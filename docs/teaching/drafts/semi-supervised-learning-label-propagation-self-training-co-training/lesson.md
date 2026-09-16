# Semi-Supervised Learning: Label Propagation, Self-Training & Co-Training

You have collected hundreds of documents, but a person has classified only a handful. Reading more documents is easy for the computer. Knowing which category each document belongs to is the expensive part. Can the documents without labels still help you build a classifier?

Sometimes. The extra documents reveal vocabulary, recurring patterns and neighborhoods. They do not tell you what those patterns mean. Semi-supervised learning uses **labeled examples together with unlabeled inputs**, connecting the two through an assumption about the task.

By the end of this lesson, you should be able to trace a label through a graph, distinguish a human label from a model's guess, run a small experiment that can reject semi-supervised learning, and explain when two representations can teach each other.

The core route is sections 1–7, followed by practice. Section 8 connects these mechanisms to deeper methods; it is an optional second pass. Refreshers on weighted averages, probabilities and matrices appear where they are needed.

## 1. What information is actually available?

Write the observed examples as $L=\{(x_i,y_i)\}$ and the unlabeled pool as $U=\{x_j\}$. An input $x$ might be a feature vector; its label $y$ is the category to predict. “Unlabeled” means that the learning procedure cannot access its target, even when we retain that target separately to audit a teaching experiment.

A probabilistic classifier estimates $p(y=c\mid x)$, a distribution over the possible classes. In binary classification, estimates $[0.2,0.8]$ lead to class 1 if we choose the larger entry. The value 0.8 is the model's confidence. It becomes a reliable frequency statement only if the probabilities are appropriately calibrated in the population where we use them.

There are two distinct destinations for the predictions.

| Task | What is available during fitting? | What must the method produce? |
|---|---|---|
| Transductive | Labels on part of a fixed collection and inputs for the whole collection | Labels for that collection's remaining items |
| Inductive | A training collection with some missing labels | A rule that can handle future inputs |

A graph algorithm naturally assigns labels to its existing nodes. Using it for a new document requires an extension: connect the new input to the training collection, or train a prediction rule using the graph. The real experiment below uses scikit-learn's inductive prediction operation; it never adds development or test inputs to the training graph.

**Inline illustration — the label ledger.** Show three compartments: six human-labeled training rows; 314 training rows whose labels are sealed; and separate development/test rows. A pseudo-label gets an outlined “model guess” badge with its round and donor. It never changes into the solid badge for a human label.

### Why unlabeled inputs cannot solve the task on their own

Imagine two dense groups of points, one near $x=-2$ and one near $x=2$. One task labels the left group 0 and the right group 1. Another task labels points by whether a hidden inspection found a defect, independently of which group they occupy. The input distribution can be identical in both tasks.

Learning the groups accurately helps the first task, once labels identify the groups. It does not reveal the hidden inspection result in the second. Extra knowledge of $p(x)$ is useful only through a relationship with $p(y\mid x)$.

Several relationships are common:

- **Local smoothness:** sufficiently similar inputs tend to have similar targets.
- **Cluster or low-density separation:** a good classification boundary tends to avoid dense regions of the input distribution.
- **Manifold structure:** relevant variation lies near a lower-dimensional surface, and the target changes smoothly along that surface.
- **Complementary views:** different representations contain useful evidence about the same target, with sufficiently different mistakes.

A curved sheet of data does not automatically imply that every connected part has one class. “The data have a manifold” and “labels are smooth on this manifold” are separate claims.

The previous Gaussian-process lesson made similarity explicit through a covariance kernel. Here we will first express it through graph edges. In both cases, a beautiful similarity picture can encode the wrong relationship.

## 2. Label propagation: let neighboring examples constrain one another

Make each example a node. Connect two nodes when their inputs are similar. A nonnegative edge weight $w_{ij}$ measures the strength of that relationship. For now the graph is undirected, so $w_{ij}=w_{ji}$, and there are no self-edges.

Consider this deliberately small, authored graph:

$$
 A\; \xleftrightarrow{\;1\;}\;B\;
 \xleftrightarrow{\;1\;}\;C\;
 \xleftrightarrow{\;1\;}\;D,
 \qquad f_A=0,\quad f_D=1.
$$

The endpoints are observed labels. The unknown value $f_B$ will be a score for class 1. **Hard clamping** means that A and D remain fixed while B and C update.

A weighted average multiplies each neighbor's value by its edge weight, adds the products, then divides by the total incident weight. At equilibrium:

$$
 f_B=\frac{f_A+f_C}{2}=\frac{f_C}{2},
 \qquad
 f_C=\frac{f_B+f_D}{2}=\frac{f_B+1}{2}.
$$

Substitute the first equation into the second: $f_C=(f_C/2+1)/2$, hence $f_C=2/3$ and $f_B=1/3$. Thresholding at 0.5 assigns B to class 0 and C to class 1. This is a consequence of the edges and boundary labels, not an additional observation about B or C.

We can also reach the answer by repeated averaging. Update all unknown nodes from the **previous** state, then restore the endpoints:

| Update | A | B | C | D |
|---|---:|---:|---:|---:|
| Initial | 0 | 0.5000 | 0.5000 | 1 |
| 1 | 0 | 0.2500 | 0.7500 | 1 |
| 2 | 0 | 0.3750 | 0.6250 | 1 |
| 3 | 0 | 0.3125 | 0.6875 | 1 |
| Equilibrium | 0 | 1/3 | 2/3 | 1 |

The intermediate values alternate around the answer. A single update is not the solution, and changing update order would produce a different intermediate trace.

### One shortcut can reverse a prediction

Add an edge from B directly to D with weight 2. Now B receives two units of influence from the class-1 anchor:

$$
 f_B=\frac{0+f_C+2}{4},\qquad f_C=\frac{f_B+1}{2}.
$$

The solution is $f_B=5/7$ and $f_C=6/7$. B changes class. Adding unlabeled examples can create similar bridges in a feature graph, so collecting more examples can alter predictions far from their immediate neighbors.

**Investigation — repair the neighborhood.** Set the B–D shortcut weight, then predict whether it makes B more or less class-1-like. Commit the prediction before running, and trace the incoming weighted contributions. Then create your own graph by editing an edge and a label, and make a new prediction. Finally restore the original chain and remove its B–C edge: B becomes 0 and C becomes 1. Add a separate pair E–F with no labeled endpoint. Its label is unknown; the interface must show the absence of evidence rather than paint both nodes as class 0.

### Why this is also an electrical circuit

Interpret each edge weight as a conductance, the inverse of resistance. Hold A at voltage 0 and D at voltage 1. At an interior node, total incoming current equals total outgoing current:

$$
 \sum_j w_{ij}(f_i-f_j)=0.
$$

Rearranging gives precisely the weighted-average equation. A strong edge resists a large disagreement between its endpoints. This analogy gives a useful diagnostic: a mislabeled boundary node can influence a whole well-connected region.

There is also a random-walk interpretation. Starting at B, repeatedly choose a neighbor with probability proportional to the edge weight. Stop on reaching a labeled node. On this graph, the chance of hitting the class-1 endpoint first is $1/3$. This probabilistic interpretation belongs to the specified walk and boundary problem; it does not establish calibration for real-world class labels. These connections are developed in [Zhu, Ghahramani and Lafferty's harmonic-function formulation](https://pages.cs.wisc.edu/~jerryzhu/pub/zgl.pdf).

### The compact matrix form

Let $W$ contain the edge weights, and let $D$ be diagonal with $D_{ii}=\sum_j w_{ij}$. The graph Laplacian is $L_g=D-W$. It measures disagreement: for a vector $f$,

$$
 f^\top L_g f=\frac12\sum_{i,j} w_{ij}(f_i-f_j)^2.
$$

The factor $1/2$ avoids counting both directions of each undirected edge twice. Minimizing this disagreement while keeping labeled values fixed produces the harmonic equations.

Partition the rows into labeled positions $\ell$ and unlabeled positions $u$:

$$
 (L_g)_{uu}f_u=W_{u\ell}y_\ell.
$$

Solve this linear system instead of explicitly forming its inverse. For a finite undirected graph with nonnegative weights, every connected component containing unknown nodes needs a labeled anchor for this boundary solution to be unique. An unanchored component admits an arbitrary constant value; no boundary label chooses that constant.

For more than two classes, use one score column per class and solve the same problem for each one. A one-hot label for class 2 in a three-class task is $[0,0,1]$: all its mass goes in the third column.

### Build the graph deliberately

A common edge weight is

$$
 w_{ij}=\exp(-\gamma\|x_i-x_j\|^2),\qquad i\ne j.
$$

Small $\gamma$ connects more distant points strongly; large $\gamma$ concentrates influence nearby. Feature units matter: a coordinate measured in thousands can dominate another measured in fractions unless the metric or scaling addresses that difference.

A nearest-neighbor graph retains only local connections. Its construction choices matter too: directed neighbor lists, their undirected union, and their mutual-neighbor intersection produce different graphs. A two-dimensional projection of four-dimensional features is a viewing aid, not proof that an edge is wrong. Inspect neighbors in the representation actually used.

## 3. Label spreading: soften the anchors without confusing the normalization

Hard propagation treats the observed labels as fixed boundary conditions. Sometimes an observed label could be wrong, or we want a regularized balance between neighbor agreement and label fidelity.

Define

$$
 S=D^{-1/2}WD^{-1/2},\qquad
 F^{(t+1)}=\alpha S F^{(t)}+(1-\alpha)Y,
 \quad 0\leq\alpha<1.
$$

$F$ now has one column per class. $Y$ contains one-hot rows at labeled nodes and **zero rows at unlabeled nodes**. Those zeros mean no injected label evidence. They are not a 50–50 class estimate.

Each update mixes propagated evidence with a fresh injection of the observed labels. Labeled rows can change, so this is soft anchoring. The normalized matrix $S$ is symmetric, but it is **not a row-stochastic transition matrix**. For our four-node chain, its row sums are approximately $[0.7071,1.2071,1.2071,0.7071]$. The random-walk matrix is $P=D^{-1}W$, whose non-isolated rows sum to one.

At a fixed point:

$$
 (I-\alpha S)F=(1-\alpha)Y.
$$

For the same endpoints and $\alpha=0.8$, the raw class-1 score at B is about 0.149652 and its class-0 score is 0.254409. Normalize that row for a distribution-shaped display:

$$
 \frac{0.149652}{0.254409+0.149652}\approx0.370370.
$$

This differs from hard propagation's $1/3$. At A, the normalized class-1 score is about 0.197531 even though its observed label is 0. Soft anchoring has an observable effect.

**Inline illustration — two layers of scores.** Keep the graph in view beside raw evidence bars and normalized readout bars. Give the bars separate labels. At $\alpha=0$, unlabeled raw rows remain all zero and their normalized class assignment is unavailable. This is “no propagation,” not the hard-clamped algorithm.

An unanchored connected component also receives no evidence and has zero scores under this update. Do not divide by a zero row sum or turn the first column of an all-zero tie into a confident label.

### What the regularizer asks for

For this symmetric graph, the fixed point minimizes

$$
 J(F)=\alpha\,\operatorname{tr}\!\left(F^\top(I-S)F\right)
       +(1-\alpha)\|F-Y\|_F^2.
$$

The first term penalizes disagreement after degree normalization; the second penalizes departure from the injected evidence. Taking the derivative and setting it to zero gives
$(I-\alpha S)F=(1-\alpha)Y$. This connects the update to an optimization problem rather than a visual blending trick. [Zhou and colleagues](https://proceedings.neurips.cc/paper_files/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf) develop the local-and-global-consistency approach.

The error obeys $E^{(t+1)}=\alpha S E^{(t)}$. Since the eigenvalues of this normalized adjacency lie in $[-1,1]$, the Euclidean error norm contracts by at most $\alpha$ per step. Increasing $\alpha$ toward one can slow convergence considerably: $0.99^{100}\approx0.366$, and $0.99^{500}\approx0.00657$. A worst-case reduction to $10^{-6}$ needs 1,375 steps at that factor. Use a residual or convergence criterion, not an animation that declares victory after an arbitrary number of frames.

### Reproduce the two different solutions

Install NumPy with `python -m pip install numpy`, save this complete program as `graph-solutions.py`, and run `python graph-solutions.py`. It constructs only the four-node anchored chain, so the inverse square-root degrees are defined and the hard system has a unique solution.

```python
import numpy as np

weight = np.array([[0., 1., 0., 0.], [1., 0., 1., 0.],
                   [0., 1., 0., 1.], [0., 0., 1., 0.]])
degree = weight.sum(axis=1)
laplacian = np.diag(degree) - weight
known, unknown = [0, 3], [1, 2]
hard = np.linalg.solve(
    laplacian[np.ix_(unknown, unknown)],
    weight[np.ix_(unknown, known)] @ np.array([0., 1.]),
)
normalized = weight / np.sqrt(degree[:, None] * degree[None, :])
evidence = np.zeros((4, 2))
evidence[0, 0] = evidence[3, 1] = 1
alpha = .8
soft = np.linalg.solve(np.eye(4) - alpha * normalized, (1-alpha) * evidence)
row_mass = soft.sum(axis=1, keepdims=True)
readout = np.divide(soft, row_mass, out=np.zeros_like(soft), where=row_mass > 0)
print("hard B,C:", np.round(hard, 6))
print("soft class-1 readout A,B,C,D:", np.round(readout[:, 1], 6))
print("has evidence:", (row_mass[:, 0] > 0).tolist())
```

The outputs are hard B,C = [0.333333, 0.666667] and soft class-1 readout = [0.197531, 0.370370, 0.629630, 0.802469], with evidence at all four nodes. A zero row in the guarded division is storage for an unavailable readout; its evidence flag must remain false. To add isolated nodes, also guard zero degrees and check component anchoring before solving the hard system.

## 4. Self-training: make a guess, record its origin, then refit

Self-training does not require an explicit graph. Start with a supervised classifier trained on $L$. Use it to propose labels for $U$. Add a selected subset of these pseudo-labels to training, then fit again.

The loop has six concrete operations:

1. Fit using observed labels and previously accepted pseudo-labels.
2. Predict class probabilities for the remaining unlabeled inputs.
3. Accept a prediction only when the selection rule allows it; a common rule is $\max_c p(c\mid x)\geq\tau$.
4. Store its predicted class, confidence, round and provenance.
5. Refit with the expanded labeled collection.
6. Stop when no candidates qualify, the pool is exhausted, the round budget is reached, or a predeclared development criterion chooses an earlier model.

The last returned model must include the final accepted batch. Returning the model fitted just before the last promotion silently drops that batch's influence.

### Watch a boundary move for the wrong reason

For a transparent mechanism, use a one-dimensional prototype classifier. Each class prototype $\mu_c$ is the average of its currently labeled inputs. Its score is

$$
 p(c\mid x)=
 \frac{\exp(-(x-\mu_c)^2)}
 {\exp(-(x-\mu_0)^2)+\exp(-(x-\mu_1)^2)}.
$$

These are deliberately chosen model scores, not calibrated probabilities. With equal distance scales, the decision boundary lies halfway between the two prototypes.

Start with observed $(-2,0)$ and $(2,1)$ and unlabeled inputs $[-1,0,1,3]$. The boundary starts at zero. With threshold 0.8, the first round accepts $-1$ as class 0, and 1 and 3 as class 1. The new prototypes are $-1.5$ and 2, so the boundary moves to 0.25. The input 0 now receives a class-0 score about 0.852 and is accepted next. Final prototypes are $-1$ and 2, giving boundary 0.5.

Replace the unlabeled input 3 with 9. The first positive pseudo-label batch pulls the class-1 prototype to 4. After the next promotion, the boundary is 1.5. A query at $x=1.25$ changes from predicted class 1 to class 0 solely because of that unlabeled point's influence. Whether the change helps depends on the query's real label, which the procedure does not know.

**Investigation — a guess changes the next guess.** Drag or edit the unlabeled points, then commit a prediction about the resulting boundary before running. Step through the old prototypes, accepted batch and newly fitted prototypes. Create an input collection of your own before revealing its trajectory. Try the null collection $[0,0]$: both scores are 0.5, no point qualifies, and the prototypes stay fixed.

This feedback is **confirmation bias**: an incorrect prediction can enter the training data and help generate more incorrect predictions. Raising the threshold can reduce promotions; it does not certify the ones that remain. A class that is initially harder to recognize can also receive fewer pseudo-labels, amplifying imbalance.

Possible responses include improving the seed coverage, changing the representation, auditing a random sample of proposed labels, weighting pseudo-labels less strongly, or stopping earlier on labeled development evidence. Each changes a specific part of the loop. None removes the need to measure the result.

### Why this is not automatically expectation–maximization

The earlier mixture-model lesson used EM with a generative model $p_\theta(x,y)$. Unlabeled inputs contribute $\log\sum_y p_\theta(x,y)$ to its likelihood, and an E-step computes latent-label responsibilities under that model.

For a purely conditional classifier, summing $p_\theta(y\mid x)$ over its classes gives 1. The corresponding unlabeled log term is $\log 1=0$; it cannot train the classifier by itself. Pseudo-labeling adds an extra assumption or objective. A thresholded, hard-label refitting loop is not automatically an EM algorithm and does not inherit EM's likelihood-ascent argument.

## 5. Co-training: let another representation provide the label

Suppose one classifier reads a web page's body and another reads the text of incoming links. A page with an unfamiliar body may still receive informative incoming links. One view can label an example whose other view is currently difficult, giving the other learner a new training pair.

This is the purpose of two views: **useful evidence that is not merely a duplicate of the same mistake**. Splitting a feature vector in half does not establish that property.

Here is an authored categorical version that exposes the transfer without hiding it inside a large classifier. In each view, the learner remembers a category-to-class rule if all training labels it has seen for that category agree. It abstains on unseen or contradictory categories.

| Row | View 1 | View 2 | Initially observed label |
|---|---|---|---|
| 0 | red | round | 0 |
| 1 | blue | square | 1 |
| 2 | red | triangle | unknown |
| 3 | green | triangle | unknown |
| 4 | orange | square | unknown |
| 5 | orange | hexagon | unknown |
| 6 | red | square | unknown |

Initially, view 1 knows red→0 and blue→1; view 2 knows round→0 and square→1.

In round 1, view 1 proposes class 0 for row 2. The recipient learns triangle→0 from **its own feature** on that row. View 2 proposes class 1 for row 4, teaching view 1 orange→1.

In round 2, triangle→0 lets view 2 teach green→0 through row 3. Orange→1 lets view 1 teach hexagon→1 through row 5. Some already implied labels are exchanged too; those confirmations are not new independent evidence.

Row 6 is different: red says 0 while square says 1. Our declared conflict rule is to defer both offers on a conflicting row. The disagreement exposes a failure of the category-consistency assumptions. It does not tell us which view is correct.

**Investigation — pass a label through the other view.** Predict what green will learn before exposing the transfers. Change row 2 from red/triangle to blue/triangle: the propagated rule for green changes from 0 to 1. Build your own paired rows and inspect each label's donor chain. Finally duplicate view 1 into view 2. Green and orange are unseen in both views, so their rows remain unresolved; two copies cannot invent complementary evidence.

### A complete, inspectable co-training procedure

The companion [categorical co-training program](cotrain-categories.py) implements exactly this exercise with Python's standard library. Save it beside your working files and run:

```text
python cotrain-categories.py
```

Its rule learner groups received labels by category, retaining a rule only when the set has one member. During a round, both learners fit first and make all proposals before either receives a new label. Each learner has its own training-label array. An accepted offer is stored only in the recipient's array, using the recipient's representation on that row. Conflicting proposals are deferred; already labeled recipient entries are preserved. Iteration stops on no offers or after eight rounds.

The printed offers have the form (row, donor view, recipient view, label):

```text
round 1 offers [(2, 1, 2, 0), (4, 2, 1, 1)] conflicts [6]
round 2 offers [(2, 2, 1, 0), (3, 2, 1, 0), (4, 1, 2, 1), (5, 1, 2, 1)] conflicts [6]
round 3 offers [(3, 1, 2, 0), (5, 2, 1, 1)] conflicts [6]
round 4 offers [] conflicts [6]
```

The final view-1 rules include green→0 and orange→1; view 2 includes triangle→0 and hexagon→1. Run the duplicate-view case and the unresolved view-1 row indices are $[3,4,5]$. The exercise teaches the information flow and its failure conditions, not the accuracy of a real document classifier.

The original [Blum–Mitchell co-training work](https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf) supplies a theoretical setting with compatible views, conditional independence given the target, a weakly useful starting predictor and noise-learnability conditions. That result is not a universal label-budget guarantee for every practical exchange loop.

Conditional independence means that, **within a fixed true class**, observing view 1’s features does not change the distribution of view 2. Merely observing agreement, splitting features, or reporting a low overall error correlation does not establish it. Practical co-training can be investigated when the theorem's assumptions are imperfect, but the experiment must earn its own conclusion.

For a real system, specify separate preprocessing for the views, a shared target definition, promotion thresholds, conflict handling, class imbalance treatment, stopping criteria and label provenance. Evaluate each view and the chosen combination on the same held-out rows. Never feed view-2 features to a model trained to interpret view-1 coordinates.

## 6. A real experiment where the supervised baseline wins

We will use four wavelet-derived measurements from the [UCI Banknote Authentication dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication), contributed by Volker Lohweg. The retained features are variance, skewness, curtosis and entropy. The source uses class codes 0 and 1; this exercise keeps those codes rather than guessing their semantic mapping.

Download the accompanying [480-row CSV](banknote-subset.csv) and keep its [provenance record](data-provenance.md). It is a deterministic subset of the 1,372-row dataset, shared under CC BY 4.0. The first 320 retained rows form the training pool; the next 80 are development data and the final 80 are the locked test set.

We reveal the first three training-pool labels from each class: six labels, with both classes deliberately represented. The other 314 training labels are unavailable to fitting. This balanced seed selection simulates an oracle; six arbitrary labels would not necessarily cover both classes.

There are also 80 development and 80 test labels. **Six training labels does not mean six labels for the whole project.** We keep this evaluation cost visible because tiny-label demonstrations can otherwise hide most of their human supervision in model selection.

Standardization fits all 320 training inputs and no development or test inputs. Every candidate uses that same label-free representation. Our logistic baseline is consequently a supervised classifier with shared unlabeled preprocessing, a controlled comparison of classifier changes rather than a strict no-unlabeled-input system.

The candidates are specified before looking at development results:

- Logistic regression with $C=1$ on the six observed labels.
- A 3-nearest-neighbor classifier on the same labels.
- RBF label spreading with $\alpha=0.2$ and $\gamma\in\{0.25,1,4\}$.
- Logistic self-training with thresholds 0.8 or 0.95 and at most ten rounds.

No model sees hidden training truth to decide its promotions. That truth is opened only for the final instructional audit of how the guesses went wrong.

### Run the experiment

Use a Python environment with NumPy and scikit-learn:

```text
python -m pip install numpy scikit-learn
python banknote-experiment.py
```

Save the following complete program as `banknote-experiment.py` beside the CSV. The checked run used Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1 and scikit-learn 1.9.1.

```python
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading

def self_train(x, initial, threshold):
    labels = initial.copy()
    history = []
    for round_number in range(1, 11):
        known = labels >= 0
        model = LogisticRegression(C=1, max_iter=500).fit(x[known], labels[known])
        remaining = np.flatnonzero(~known)
        if remaining.size == 0:
            break
        probability = model.predict_proba(x[remaining])
        accept = probability.max(axis=1) >= threshold
        rows = remaining[accept]
        guessed = model.classes_[probability[accept].argmax(axis=1)]
        history.append((rows.copy(), guessed.copy()))
        if rows.size == 0:
            break
        labels[rows] = guessed
    known = labels >= 0
    model = LogisticRegression(C=1, max_iter=500).fit(x[known], labels[known])
    return model, history

table = np.genfromtxt(
    Path(__file__).with_name("banknote-subset.csv"),
    delimiter=",", names=True, dtype=None, encoding="utf-8",
)
raw = np.column_stack([table[name] for name in
                       ["variance", "skewness", "curtosis", "entropy"]])
truth = table["class"]
assert len(table) == 480
pool, dev, test = np.arange(320), np.arange(320, 400), np.arange(400, 480)
x = StandardScaler().fit(raw[pool]).transform(raw)
seed = np.concatenate([np.flatnonzero(truth[pool] == c)[:3] for c in [0, 1]])
initial = np.full(320, -1)
initial[seed] = truth[seed]
models = {
    "supervised_lr": LogisticRegression(C=1, max_iter=500).fit(x[seed], truth[seed]),
    "supervised_knn": KNeighborsClassifier(3).fit(x[seed], truth[seed]),
}
histories = {}
for gamma in [.25, 1., 4.]:
    models[f"spreading_{gamma}"] = LabelSpreading(
        kernel="rbf", gamma=gamma, alpha=.2, max_iter=1000, tol=1e-6
    ).fit(x[pool], initial)
for threshold in [.8, .95]:
    model, history = self_train(x[pool], initial, threshold)
    models[f"self_{threshold}"] = model
    histories[threshold] = history

correct = {}
for name, model in models.items():
    correct[name] = int(np.sum(model.predict(x[dev]) == truth[dev]))
    print(name, correct[name], "/ 80")
# Dictionary order resolves ties in favor of the simpler baseline, as predeclared.
selected = max(correct, key=correct.get)
test_correct = int(np.sum(models[selected].predict(x[test]) == truth[test]))
print("selected", selected, "test", test_correct, "/ 80")
# Offline explanation only: these targets never enter self_train.
for threshold, history in histories.items():
    print("threshold", threshold, "accepted", [len(rows) for rows, _ in history])
    print("wrong in offline audit",
          [int(np.sum(guessed != truth[rows])) for rows, guessed in history])
```

The development results are:

| Candidate | Correct / 80 | Accuracy |
|---|---:|---:|
| Logistic baseline | 72 | 0.9000 |
| 3-neighbor baseline | 63 | 0.7875 |
| Spreading, $\gamma=0.25$ | 61 | 0.7625 |
| Spreading, $\gamma=1$ | 59 | 0.7375 |
| Spreading, $\gamma=4$ | 65 | 0.8125 |
| Self-training, $\tau=0.8$ | 59 | 0.7375 |
| Self-training, $\tau=0.95$ | 72 | 0.9000 |

The 0.95 self-training run promotes no examples, so it is the logistic baseline again. The predeclared tie rule selects that simpler baseline. It correctly predicts **72 of the 80 locked test rows**. We do not refit on development labels, preserving the six-label training comparison.

The 0.8 run tells a more revealing story:

| Promotion round | Newly accepted | Wrong in the offline audit |
|---|---:|---:|
| 1 | 48 | 0 |
| 2 | 106 | 12 |
| 3 | 60 | 28 |
| 4 | 18 | 14 |
| 5 | 11 | 10 |
| 6 | 6 | 6 |
| 7 | 3 | 3 |
| 8 | 0 | 0 |

Its first batch looks excellent. Later guesses become increasingly unreliable as the training labels feed back into the decision boundary. It accepts 252 pseudo-labels, including 73 wrong ones, and leaves 62 training examples unlabeled. Increasing the training set's apparent size has not increased its trustworthy information.

**Inline illustration — the promotion audit.** Use two stacked counts per round, correct and wrong, with a separate trace for total accepted labels. The y-axis counts examples; there is no invented performance curve between measured rounds. Keep the caption “hidden truth opened after training for explanation.” This information would require fresh labeling in a genuinely unlabeled application.

The appropriate conclusion is narrow: under this subset, seed policy, representation and candidate set, the measured evidence selects the baseline. It does not establish that SSL always fails on banknotes, or that a differently tuned graph cannot work. The useful action is to retain the baseline and investigate a specific representation or label-coverage hypothesis before launching another experiment.

## 7. Choose the method by the information it can use

| Available structure | Candidate to investigate | First thing to inspect |
|---|---|---|
| Meaningful pairwise similarities and a fixed collection | Graph propagation or spreading | Neighbor quality, cross-class shortcuts and unanchored components |
| A reasonable initial classifier and many relevant inputs | Self-training | Pseudo-label correctness by class and round, with observed-label comparison |
| Two representations with complementary evidence | Co-training | View-specific errors, transfer provenance and conflicts |
| Label-preserving transformations | Consistency-based learning | Whether each transformation preserves this task's target |

For an annotation team, a graph can expose islands with no labeled anchor and suggest where one additional label might matter. For linked biological entities, neighborhood information can suggest a function, but an interaction edge is not automatically evidence for the same function. For paired audio and visual observations, agreement can be informative, yet common background artifacts may make both views wrong together. These are questions to test, not automatic reasons to adopt SSL.

### Keep the compute proportional to the collection

A dense graph on $n$ nodes stores $n^2$ weights. At 100,000 nodes, one float64 weight matrix alone takes $8\times10^{10}$ bytes, about 80 GB before scores, copies or overhead. That storage calculation follows from dimensions; it is not a timing benchmark.

A sparse graph with $E$ stored directed edges takes roughly $O(E+n)$ graph storage plus $O(nC)$ scores for $C$ classes. The graph multiplication costs $O(EC)$, and adding the label evidence and updating all scores adds $O(nC)$. Constructing the nearest-neighbor graph can itself be expensive, especially in high dimensions, and symmetrizing it can increase the number of stored edges. Count indices and row pointers as well as weight values.

Self-training's cost is the sum of fitting and prediction over its rounds, not one supervised fit. Co-training can require two such fitting sequences. Cache representations that do not change; avoid repeated full graph construction when only a display changes; keep the labeled and pseudo-labeled provenance separate from the numerical arrays.

### Put the common failure checks in one place

Ask whether training and unlabeled examples share the relevant target space; whether all important classes appear among the seeds; whether the graph or augmentation encodes the intended similarity; and whether the development labels are sufficient for the comparison you are making.

Keep test inputs out of inductive training preprocessing, even when their labels are hidden. If you intentionally use all inputs for a transductive problem, state that protocol and evaluate that task. Do not use hidden benchmark labels for stopping, then describe the loop as unlabeled.

Class-balanced promotion can be a hypothesis when one class is being ignored. It also imposes a selection policy, and an equal quota may be inappropriate for unequal real prevalence. A high threshold, a nice cluster plot and agreement between models each provide a different kind of evidence; none alone proves correctness.

[Oliver and colleagues' evaluation study](https://arxiv.org/pdf/1804.09170) is a useful reminder to compare underlying models fairly, account for validation labels and investigate mismatched unlabeled data.

## 8. Optional depth: what connects these methods to later learning systems?

### Generative models and low-density boundaries

A generative mixture can use all inputs to estimate where components lie, while observed labels help connect components to classes. The benefit depends on whether the model's component assumptions match the task. Better input-density fit does not imply better classification.

A transductive support-vector method instead searches for labels and a large-margin boundary jointly, encouraging separation in low-density regions. The discrete unknown labels make this a harder optimization problem than ordinary supervised convex SVM fitting. Entropy regularization encourages confident predictions on unlabeled inputs; it needs safeguards because confidently assigning everything to one class can satisfy a confidence objective poorly aligned with the task.

Manifold regularization combines a supervised loss, a function-complexity penalty and graph disagreement. Schematically:

$$
 \frac1{|L|}\sum_{i\in L}\ell(y_i,f(x_i))
 +\lambda_A\|f\|_{\mathcal H}^2
 +\lambda_I f(X)^\top L_g f(X).
$$

The first term learns from observed labels, the second controls the prediction rule and the third lets the input geometry constrain it. The kernel space connects back to the GP lesson; the graph term connects to section 2. Constants depend on the formulation, so preserve them when reproducing a specific algorithm.

### Deep consistency and teacher–student learning

In [FixMatch](https://arxiv.org/pdf/2001.07685), a weakly augmented input supplies a pseudo-label; a strongly augmented version is trained toward it:

$$
 q=p_\theta(\cdot\mid a_{\rm weak}(x)),\quad
 \hat y=\arg\max_c q_c,\quad
 \ell_u=\mathbf1[\max q\geq\tau]\,
       [-\log p_\theta(\hat y\mid a_{\rm strong}(x))].
$$

Treat the selected target and acceptance decision as fixed for that gradient update. Average over the specified unlabeled batch and combine with supervised loss using a weight $\lambda_u$. A horizontal flip may preserve an animal category but alter the interpretation of a character. The transformation is part of the modeling assumption.

[Noisy Student](https://arxiv.org/pdf/1911.04252) trains a teacher, generates pseudo-labels, then trains a student with input and model noise before optionally repeating the process. The student can have equal or greater capacity, and the noise introduces consistency pressure; a pseudo-label still has an origin and can still be wrong.

Self-supervised representation learning is related but distinct: its training targets can be constructed from inputs, such as masked pieces or paired views. A later classifier may use that representation with scarce human labels. Compare such a representation baseline when appropriate rather than assuming all useful unlabeled learning must happen through label propagation.

### Learning guarantees need the relationship to be stated

If two possible worlds have the same input distribution but different target rules, unlabeled inputs alone cannot distinguish them. A guarantee must restrict that ambiguity through a hypothesis class, compatibility condition, graph assumption or other stated relationship. The later PAC and generalization lessons formalize what can be inferred from a limited sample; they do not turn “more unlabeled data” into an assumption-free promise.

## 9. Practice: produce a decision and explain its evidence

Try each question before opening the hint or solution.

### A. Change the graph

In the A–B–C–D chain, use edge weights A–B=2, B–C=1, C–D=1, with endpoint labels 0 and 1. Compute B and C. Which endpoint gained influence?

<details><summary>Hint</summary>

The new average is $f_B=f_C/3$. The other interior equation remains $f_C=(f_B+1)/2$.

</details>
<details><summary>Solution</summary>

Substitution gives $f_C=(f_C/3+1)/2$, so $f_C=3/5$ and $f_B=1/5$. The stronger connection to the class-0 endpoint pulls both values downward from $1/3,2/3$.

</details>

### B. Diagnose a normalization error

A teammate says every row of $S=D^{-1/2}WD^{-1/2}$ is a probability distribution and initializes all unknown rows of $Y$ to $[0.5,0.5]$. What two meanings have been mixed up?

<details><summary>Hint</summary>

Check the four-node chain's row sums and distinguish injected evidence from normalized display scores.

</details>
<details><summary>Solution</summary>

$S$ is a symmetric normalized adjacency, not generally a transition matrix. $P=D^{-1}W$ is the row-stochastic walk matrix on non-isolated nodes. Standard spreading uses zero unknown rows in $Y$; adding uniform rows injects additional evidence and changes the objective. Row-normalized output is a separate readout, unavailable when a row has zero mass.

</details>

### C. Identify what a pseudo-label can move

Use the prototype classifier with observed $(-3,0),(3,1)$, unlabeled $[-2,2,8]$, and threshold 0.8. What is the boundary after the first accepted batch? What must be checked before claiming improvement?

<details><summary>Hint</summary>

All three inputs have strong initial distance-based scores. Recompute each class mean using the original example as well as its accepted guesses.

</details>
<details><summary>Solution</summary>

The new means are $(-3-2)/2=-2.5$ and $(3+2+8)/3=13/3$. The boundary is $(-2.5+13/3)/2=11/12$. It moved right from zero. Improvement requires observed-label evaluation on the intended prediction population; the number accepted and their model confidence are insufficient.

</details>

### D. Break the information bridge

In the categorical co-training table, replace row 2 with violet/triangle while leaving all other rows unchanged. Which new rules can still be learned? Which chain is broken?

<details><summary>Hint</summary>

Neither learner initially recognizes violet or triangle. The orange/square row still has a known second-view category.

</details>
<details><summary>Solution</summary>

Square→1 still teaches orange→1, which teaches hexagon→1. No observed or inferred rule reaches triangle, violet or green, so the class-0 transfer through rows 2 and 3 cannot start. This is an absence of a bridge, not evidence that the unresolved categories belong to class 1.

</details>

### E. Read the banknote result

The 0.8 run's first 48 pseudo-labels are all correct in the offline audit. Why does that not justify continuing to exhaustion? What did the 0.95 result establish?

<details><summary>Hint</summary>

Later predictions come from a different fitted model, and no acceptance is a legitimate outcome.

</details>
<details><summary>Solution</summary>

Refitting changes the boundary and the remaining pool is not the same collection as the first accepted batch. Later correctness can deteriorate, as the checked counts show. At 0.95 no points passed the rule, so this candidate reproduced the initial classifier. It showed no benefit from pseudo-labeling under that setting; it did not show that all high-confidence pseudo-labels are safe.

</details>

### F. Audit the learning budget

A report says “only six labels,” but uses six seed labels, 80 development labels and 80 test labels. It chooses its graph bandwidth after evaluating all three on the test labels. Rewrite the protocol.

<details><summary>Hint</summary>

Separate fitting, selection and final reporting, and count supervision outside the training loop.

</details>
<details><summary>Solution</summary>

Report six fitting labels plus 160 evaluation labels. Choose graph bandwidth and other candidates on development data with a predeclared rule. Evaluate the selected procedure once on the locked test set. If earlier test comparisons have already influenced decisions, that set has served as development data; reserve fresh evaluation data or report the limitation explicitly rather than relabeling the old result.

</details>

### G. Test the claimed EM explanation

Someone adds $\sum_{x\in U}\log\sum_c p_\theta(c\mid x)$ to logistic regression and says unlabeled examples will improve its parameters. Compute the new term and describe a mechanism that would actually introduce information.

<details><summary>Hint</summary>

Sum a normalized conditional distribution over every class.

</details>
<details><summary>Solution</summary>

Each inner sum is one, so the added term is zero. A generative model for $p_\theta(x,c)$, a graph smoothness penalty, a pseudo-label objective or a label-preserving consistency constraint could introduce an additional relationship. Its assumptions and evaluation must then be stated.

</details>

### H. A small independent investigation

Using the CSV, reserve the same locked test rows. On a new development-only experiment, vary the number of observed training labels while retaining shared preprocessing and the same logistic baseline for each budget. Record how seeds are selected, pseudo-label counts and development accuracy. Form a hypothesis about why the threshold-0.8 run deteriorates before changing the method.

<details><summary>Hint</summary>

Changing the seed set and the algorithm simultaneously makes attribution difficult. Never use hidden training truth to select individual promotions.

</details>
<details><summary>Solution approach</summary>

Choose deterministic, documented seed sets and compare supervised and self-trained versions within each set. A useful hypothesis is that a small seed set misrepresents part of a class, making later confident promotions unreliable. More representative observed labels may help; they may also change the threshold at which promotions occur. Report the measured result even if the hypothesis fails. These changed experiments have no preclaimed numeric answer, and the original final-test result does not validate them.

</details>

## 10. Another route through the subject

- [Zhu's literature survey, July 2008 version](https://pages.cs.wisc.edu/~jerryzhu/pub/ssl_survey.pdf): begin with the FAQ and then sections 3, 4 and 6. It provides a broad classical map and a useful discussion of when assumptions can fail; read it as a historical foundation.
- [CMU graph semi-supervised learning notes](https://www.cs.cmu.edu/~wcohen/10-605/notes/graph-ssl.pdf): an alternative mathematical route through graph objectives and propagation. Use it after the four-node calculation if you prefer lecture notes to a research paper.
- [CMU 10-601 lecture 19 video](https://www.youtube.com/watch?v=gnNLjX50F7U), with its [lecture slides](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/19_ssl_03-30-2015.pdf): a spoken route through transductive SVMs, co-training and graph methods. The co-training and graph portions of the slides are particularly useful after sections 2–5.
- [scikit-learn's semi-supervised guide](https://scikit-learn.org/stable/modules/semi_supervised.html) and [LabelSpreading API](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html): implementation vocabulary, unlabeled encoding, parameters and the distinction between fitted-pool transduction and new-input prediction.
- [Blum and Mitchell's co-training paper](https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf): read the two-view setup before its theorem. It explains why representation assumptions are more demanding than simply having two models.
- [Chapelle, Schölkopf and Zien's edited book](https://academic.oup.com/mit-press-scholarship-online/book/41571): the contents map generative, low-density, graph, representation and practical families. This is a deeper reference, with full-text access depending on availability.

The next module topic is **Active Learning**. Instead of allowing a model to supply every new label, it asks which examples a human or other labeling oracle should label next. The graph's unanchored island, co-training's conflict and self-training's uncertain boundary are useful reasons to investigate a query; they are not yet guarantees that a query will be valuable.
