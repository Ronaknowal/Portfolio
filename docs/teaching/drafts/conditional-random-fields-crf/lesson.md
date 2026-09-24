# Conditional Random Fields (CRF)

Suppose a reading tool should highlight a person's whole name. In “Maya Chen joined Cedar Labs,” recognizing *Chen* separately from *Maya* is an awkward decision: the first name tells us something about the next word's label. Yet blindly joining every pair of capitalized words would confuse people, organizations and sentence beginnings. We need evidence from the words **and** preferences about how neighboring labels fit together.

A conditional random field assigns a score to each complete labeling, then turns those scores into a probability distribution. A good labeling can win because several modest pieces of evidence support one another. The model's central question is: **given this observed input, which combinations of output labels make sense together?**

**First pass:** read §§1–6, trace the two-position example, and do practices 1–4. Then run the real text experiment in §7 and inspect the errors. Sections 8–9 are deeper branches on neural outputs, generalized structures and conditional sampling; return to them after you can distinguish a best path from a marginal probability. Practice 5 connects the whole workflow. Allow about 50 minutes for the reading and a separate session for code and practice.

## 1. From independent decisions to a structured output

The preceding [Bayesian Networks & Causal Graphical Models](/learn/path/full-curriculum/bayesian-networks-causal-graphical-models?module=classical-ml) describes probability factorizations and the extra assumptions needed for causal interpretation. Here the graph organizes **compatible labels**. An edge between labels carries a statistical interaction, not a causal claim about one word producing another.

The [HMM lesson](/learn/path/full-curriculum/hidden-markov-models-hmm?module=classical-ml) supplies an especially useful bridge. An HMM defines a joint distribution over observations and hidden states. For labeling, we can condition that distribution on the observed sequence. A CRF starts directly with the conditional distribution. We will reuse the HMM's dynamic-programming pattern while changing what the numbers represent.

For a sentence of length \(T\), let \(x=(x_1,\ldots,x_T)\) be its observed words and let \(y=(y_1,\ldots,y_T)\) be its unknown labels. Each label comes from a set of \(K\) possibilities. For named entities, one useful scheme is:

| Label | Meaning |
| --- | --- |
| `B-PER` | Beginning of a person span |
| `I-PER` | Another token inside that person span |
| `B-ORG`, `I-ORG` | Beginning/inside an organization |
| `O` | Outside the annotated entities |

The `B` and `I` distinguish two adjacent people from one long person name. “Maya Chen” can be `B-PER I-PER`; “Maya and Chen” has an intervening `O`. This annotation convention is called **BIO**. We will return to legal BIO paths in §6; our first arithmetic example uses only labels `A` and `B` so the mechanics stay visible.

**Inline figure — words, labels and factors.** Show the observed word strip above a row of circular label variables. Small square factors connect neighboring label circles; short vertical links attach word-derived evidence to each circle. Place an explicit “all words are observed” bracket over the input. A factor can inspect the entire input even though it joins only adjacent output labels. This distinction is what keeps rich input context compatible with a tractable chain.

A feature is simply a measured property used in a score: “this word ends in `ing`,” “this token begins with a capital,” or “these adjacent labels are `B-PER I-PER`.” Two input features may overlap or be correlated. We are not multiplying independent probabilities for those features. We are adding learned contributions to a score.

The operational boundary is **information available when the task runs**. A complete-document tagger may use the next word. A live tagger that must decide before the next word arrives needs a causal feature window or an explicit delay. Gold labels of neighboring words are targets, not free input features.

## 2. Score first, normalize once

Let \(e_t(j)\) be the score for assigning label \(j\) at position \(t\), based on the observed input. Let \(A_{ij}\) score the transition from previous label \(i\) to current label \(j\). These are real-valued **log scores**, not transition probabilities. For now, the start and end scores are zero:

\[
s(x,y)=\sum_{t=1}^{T}e_t(y_t)+\sum_{t=2}^{T}A_{y_{t-1},y_t}.
\]

Add the selected entries along a labeling to obtain its score. Exponentiation turns it into a positive mass. Dividing by the total mass makes a probability:

\[
p(y\mid x)=\frac{\exp s(x,y)}{Z(x)},\qquad
Z(x)=\sum_{y'\in\mathcal Y^T}\exp s(x,y').
\]

The **partition function** \(Z(x)\) is the normalization total for this input. It does not depend on which candidate labeling we are asking about, because every candidate uses the same denominator. It generally changes when the input changes.

Here is a complete two-position model. To make hand calculation simple, display the positive factors \(\exp e\) and \(\exp A\):

| Input contribution | A | B |
| --- | ---: | ---: |
| Position 1 | 3 | 1 |
| Position 2 | 1 | 2 |

| Pair factor: previous → current | A | B |
| --- | ---: | ---: |
| A | 1 | 4 |
| B | 1 | 1 |

The pair factor rewards `A → B` fourfold while holding the other factors fixed. The log score adds \(\log 4\); the mass multiplies by 4. Enumerate every labeling:

| Path | Multiplication | Mass | Probability |
| --- | --- | ---: | ---: |
| AA | \(3\times1\times1\) | 3 | 0.100000 |
| AB | \(3\times4\times2\) | 24 | 0.800000 |
| BA | \(1\times1\times1\) | 1 | 0.033333 |
| BB | \(1\times1\times2\) | 2 | 0.066667 |

The total is 30, so `AB` has probability \(24/30=0.8\). The score of `AB` is \(\log 24\), whereas \(\log Z=\log 30\approx3.401197\). A high score becomes meaningful as a probability only after comparison with the alternatives.

**Inline figure — a four-path ledger.** Align each path's two label chips with its selected input and pair factors, then show its mass as a horizontal bar on a common 0–30 scale. The four bars feed one denominator of 30. A second view groups `AA+AB` and `BA+BB`; it immediately shows why the first label has probability 0.9 of being A even though the best complete path has probability 0.8.

Two useful invariances follow directly. Adding the same constant \(c\) to every complete path score multiplies every mass and the denominator by \(e^c\), leaving all probabilities unchanged. Setting **all** scores to zero gives a uniform distribution on the allowed paths. Without constraints there are \(K^T\) such paths; with constraints count the legal paths instead.

This model includes an independent classifier as a special case. If every pair score is zero, each pair factor is one and

\[
Z(x)=\prod_t\sum_j e^{e_t(j)},\qquad
p(y\mid x)=\prod_t\operatorname{softmax}(e_t)_{y_t}.
\]

The factorization follows by distributing the sum over all label combinations. Removing output interactions therefore changes the model family without requiring different input features.

## 3. Sum paths or choose one: two different questions

With 10 labels and 20 positions, enumeration requires \(10^{20}\) paths. A chain lets us reuse work. Any prefix ending in A can be extended by the same next-step factors, regardless of how that prefix reached A.

### Forward messages collect all compatible prefixes

Define \(\alpha_t(j)\) as the **total unnormalized mass** of all prefixes ending in label \(j\) at position \(t\). Then

\[
\alpha_1(j)=e^{e_1(j)},\qquad
\alpha_t(j)=e^{e_t(j)}\sum_i\alpha_{t-1}(i)e^{A_{ij}},\qquad
Z=\sum_j\alpha_T(j).
\]

For the two-position example, the initial masses are \([3,1]\). The next masses are:

\[
\alpha_2(A)=1(3\cdot1+1\cdot1)=4,\qquad
\alpha_2(B)=2(3\cdot4+1\cdot1)=26.
\]

We recover 30 without separately listing all four complete paths. Notice that 26 includes both `AB` and `BB`. Replacing it by 24 would discard a valid path: that is a **maximum**, not a forward sum.

### Viterbi retains the best prefix

To find the most probable complete labeling, the common denominator can be ignored. Keep the best log score \(\delta_t(j)\) ending at each label:

\[
\delta_t(j)=e_t(j)+\max_i[\delta_{t-1}(i)+A_{ij}].
\]

Store which predecessor attained the maximum. After choosing the best final label, follow these **backpointers** backward. In our example, the best masses ending at A and B are 3 and 24; the winning path is `AB`. The sum-product and max-product computations share the same trellis but answer different questions.

**Investigation — edit a trellis.** Keep the pair factors visible and edit an input factor. Record whether the independent classifier and chain will choose the same complete path. Then compare their paths, the sum of all paths, and the best path's share. In one suggested contrast, change the second position's A factor from 1 to 6: the independent classifier favors `AA`, while the chain still favors `AB` because 24 exceeds 18. You will also build an input where the pair reward cannot change the winner.

### Backward messages bring later evidence to earlier labels

Define \(\beta_t(i)\) as the total suffix mass after position \(t\), assuming its label is \(i\). The current position's input factor is already in \(\alpha_t\), so do not include it again:

\[
\beta_T(j)=1,\qquad
\beta_t(i)=\sum_j e^{A_{ij}+e_{t+1}(j)}\beta_{t+1}(j).
\]

In the running example, \(\beta_1(A)=1+8=9\), while \(\beta_1(B)=1+2=3\). Multiply prefix and suffix, then divide by the full total:

\[
p(y_t=j\mid x)=\frac{\alpha_t(j)\beta_t(j)}{Z}.
\]

Thus \(p(y_1=A\mid x)=3\cdot9/30=0.9\). This is the same 0.9 obtained by grouping the full path ledger. The two calculations agree because both sum exactly the paths whose first label is A; dynamic programming only changes how the addition is organized.

For neighboring labels, the edge marginal is

\[
p(y_{t-1}=i,y_t=j\mid x)=
\frac{\alpha_{t-1}(i)e^{A_{ij}+e_t(j)}\beta_t(j)}{Z}.
\]

The `A → B` edge marginal is 0.8. Edge marginals will supply the expected transition counts used to train the model.

### The most likely labels need not form the most likely path

Consider a different two-position distribution with probabilities `AA=.35`, `AB=.34`, `BA=.01`, `BB=.30`. The best full path is `AA`. But the first marginal favors A with probability .69, and the second favors B with probability .64. Independent marginal decisions therefore give `AB`.

This is a decision-loss distinction. Maximizing a complete path's probability minimizes expected **whole-sequence 0–1 loss**: either every label is right or the answer counts as wrong. Choosing each marginal mode minimizes expected **number of wrong tokens** when choices are unconstrained. Additional legality constraints require constrained risk minimization. Decide what constitutes an error before deciding which output to extract.

## 4. How training changes the scores

We have chosen factors by hand. Training learns their weights from labeled examples. Write the total feature vector of a sequence as \(F(x,y)\), obtained by summing local feature vectors. With weights \(w\), the score is \(w^\top F(x,y)\).

For one observed input and gold labeling, the log-likelihood is

\[
\ell(w)=w^\top F(x,y_{\rm gold})-\log Z_w(x).
\]

Increasing the gold score helps, but increasing **every** score equally achieves nothing. The normalizer accounts for that competition. Differentiating gives

\[
\nabla_w\ell=F(x,y_{\rm gold})-
\mathbb E_{p_w(y\mid x)}[F(x,y)].
\]

Read this as **observed count minus expected count**. Suppose the only adjustable weight is the `A → B` log score and the gold path in our example is `AB`. Its observed count is 1; the model expects 0.8 occurrences. The log-likelihood gradient is therefore +0.2. A small ascent update increases that weight and shifts probability toward paths containing the edge. If the gold path were `AA`, the same weight's gradient would be −0.8.

The expectation is not computed by sampling complete paths. Sum each local feature value against the relevant node or edge marginal. Forward–backward therefore serves both prediction and training.

For \(N\) sequences, our real experiment minimizes the average negative log-likelihood plus \(\lambda\|w\|^2/2\). Its gradient is average **expected minus observed** counts plus \(\lambda w\). With regularization, the optimum satisfies a balance against shrinkage; exact unregularized moment matching no longer applies.

**Inline figure — count balance.** Use two aligned counters for the observed and model-expected `A → B` counts. Show the signed difference and the direction of the weight update. Switching the gold path changes the observed count while holding the current model distribution fixed. This makes training a correction of probability allocation, not an unexplained optimizer call.

For fixed features, fully observed labels and a finite-dimensional linear score, the negative log-likelihood is convex: the Hessian of \(\log Z\) is the covariance matrix of the features, hence positive semidefinite. Positive L2 regularization makes this parameter objective strictly convex. An optimizer still needs an appropriate step or line search and a checked stopping condition. Neural feature learning or marginalization over latent labels changes the objective and can remove convexity. These are the conditions attached to the result, rather than properties of everything called a CRF.

### A stable program for the exact chain operations

Install the packages in your own Python environment with `python -m pip install numpy scipy scikit-learn`. Save this block as `crf_example.py`; the real-data block in §7 continues the same file. Python 3.12.14, NumPy 2.3.5 and scikit-learn 1.9.1 were the author calculation environment. The final website execution and packaging pass remains separate.

```python
import numpy as np

def infer(emissions, transitions):
    length, labels = emissions.shape
    forward = np.empty_like(emissions)
    backward = np.zeros_like(emissions)
    forward[0] = emissions[0]
    for t in range(1, length):
        forward[t] = emissions[t] + np.logaddexp.reduce(
            forward[t-1, :, None] + transitions, axis=0)
    log_partition = np.logaddexp.reduce(forward[-1])
    for t in range(length-2, -1, -1):
        backward[t] = np.logaddexp.reduce(
            transitions + emissions[t+1] + backward[t+1], axis=1)
    nodes = np.exp(forward + backward - log_partition)
    edges = np.array([
        np.exp(forward[t-1, :, None] + transitions
               + emissions[t] + backward[t] - log_partition)
        for t in range(1, length)])
    best = emissions[0].copy()
    parents = []
    for t in range(1, length):
        candidates = best[:, None] + transitions
        parents.append(candidates.argmax(axis=0))
        best = candidates.max(axis=0) + emissions[t]
    path = [int(best.argmax())]
    for parent in reversed(parents):
        path.append(int(parent[path[-1]]))
    return float(log_partition), nodes, edges, path[::-1]

emissions = np.log([[3., 1.], [1., 2.]])
transitions = np.log([[1., 4.], [1., 1.]])
logz, nodes, edges, path = infer(emissions, transitions)
print(round(np.exp(logz), 6))
print(np.round(nodes, 6))
print(path)
```

Run `python crf_example.py`. The author arithmetic produced:

```text
30.0
[[0.9      0.1     ]
 [0.133333 0.866667]]
[0, 1]
```

Label indices 0 and 1 mean A and B. Each row of `nodes` sums to one. Transition rows are **previous labels**, columns are **current labels**; that convention is why `axis=0` combines predecessors in the forward pass.

Log-space summation avoids multiplying a long string of tiny factors. For real numbers \(a_i\),
\(\operatorname{LSE}(a)=m+\log\sum_i e^{a_i-m}\), where \(m=\max_i a_i\).
`logaddexp.reduce` performs this stable kind of combination. The program accepts finite log scores and sequences of length at least two for its edge-array shape. A constrained version additionally handles impossible paths and all-\(-\infty\) reductions explicitly. Do not exponentiate huge raw scores before the forward pass; after subtracting \(\log Z\), the marginal log probabilities are at most zero apart from rounding.

## 5. Why normalize globally? A label-bias experiment

A maximum entropy Markov model, or MEMM, uses a locally normalized conditional distribution at each step. It can incorporate rich observed features, but every state's outgoing probabilities must add to one. The distinction is visible in a tiny constructed model.

At the first position, branches A and B receive probabilities .5 each. Each has exactly one permitted continuation. Later input evidence gives A's continuation an unnormalized compatibility of .01 and B's a compatibility of 1.

With **local normalization**, A's only exit has probability \(.01/.01=1\); B's only exit also has probability 1. Both complete paths retain probability .5. The later evidence cancels inside each branch's private denominator.

With **global normalization**, the complete path masses are \(.5\cdot.01=.005\) and \(.5\cdot1=.5\). Their probabilities are \(1/101\approx.009901\) and \(100/101\approx.990099\). The branches now compete using their compatibility with the later observation.

**Investigation — two branches, two normalizers.** Drag the later compatibility ratio, record which branch will gain probability under each normalization rule, and apply the edit. Keep both denominator calculations visible. At equal compatibility, both rules return .5/.5; this null case distinguishes the mechanism from a claim that the methods must always disagree.

The classic label-bias issue concerns the information path and local normalization in that model. Feeding later input into earlier local classifiers can change what they know; an HMM's locally normalized *generative* factors also do not have the same information structure. Global normalization gives the CRF a direct way to compare full paths, not a theorem that every CRF outperforms every directed or independently normalized alternative. The original problem and these distinctions are developed in the [Sutton–McCallum tutorial, §6.1.3](https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf).

## 6. A preference is different from a rule

Return to BIO labels. A learned score of −5 for `O → I-PER` discourages that transition. An input score of +20 for `I-PER` can overwhelm it. A finite penalty therefore expresses a preference, not impossibility.

To enforce BIO legality, define the permitted path set before inference. At the first position, `I-X` is illegal. Later, `I-X` may follow only `B-X` or `I-X` of the same entity type. Use a zero-valued factor for a forbidden transition, equivalently a log score of \(-\infty\). Include the same legality rules in the training partition function when training a distribution over legal paths. Merely constraining the decoder after unconstrained training is a different training/prediction combination.

For a one-position example with labels `[O, B-PER, I-PER]` and scores `[0, 0, 10]`, unconstrained decoding chooses `I-PER`. Masking the illegal start removes that label, leaving a tie between the two legal alternatives. The model needs more evidence to choose between those alternatives; a legality rule cannot supply missing semantics.

**Inline figure — legal span automaton.** Draw `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG` states and a separate start marker. Highlight the incoming edges to a selected `I-X` label rather than drawing every possible edge at once. Solid means allowed; a crossed-out attempted edge shows the precise failed rule. Pair it with one valid and one invalid three-token strip.

Library switches need their own interpretation. In `sklearn-crfsuite`, `all_possible_transitions=True` creates feature parameters for every label pair, including pairs absent from training. Setting it to false controls **which transition features are generated**; it is not a BIO legality mask. Inspect a library's actual support for constraints rather than inferring it from an option name. The [API reference](https://sklearn-crfsuite.readthedocs.io/en/latest/api.html) describes these feature-generation options.

The same modeling idea can help with nonlinguistic structured output. For an original teaching scenario, imagine a scanned form whose lines are labeled heading, value or footer. A pair preference can reward a value after a heading; a deterministic rule can require a signature field before a finalized status. The first is statistical evidence; the second defines allowed outputs. Whether the application truly requires a hard rule is a product and annotation decision, not something to learn from a few repeated examples.

## 7. Does output structure help on real text?

The [Universal Dependencies English EWT treebank](https://universaldependencies.org/treebanks/en_ewt/index.html) supplies linguistic annotations of real web text. Its part-of-speech tags describe syntactic roles. Our offline file, [ewt-sequences.json](ewt-sequences.json), preserves token strings, original tags, sentence IDs and official split membership from release `r2.16`; its derivative data is shared under CC BY-SA 4.0. [Data provenance](data-provenance.md) records the extraction and attribution.

Our deliberately small question is: **with the same word features and training sentences, does adding neighboring-label scores improve coarse tagging on held-out sentences?** Collapse `NOUN/PROPN` to NOUN, `VERB/AUX` to VERB, and all other tags to OTHER. OTHER is a teaching aggregation, not a universal linguistic category.

We retain the first 120 training, 40 development and 40 test sentences of length 3–15 from their respective official files. This makes the experiment quick and inspectable; it is not a random or representative benchmark sample. The data units remain whole sentences. Vocabulary construction and parameter fitting use training text only. We compare two prespecified models with the same features and regularization: independent per-token scores and a chain with nine extra transition scores. Development token accuracy chooses the model; the final test summarizes that choice.

The features are lowercased current word, two-character suffix, capitalization, and previous/next word, plus a bias. The whole sentence is observed in this offline task. Context words are inputs; the gold neighboring tags are never supplied to the feature extractor.

Append this complete block to `crf_example.py` and place `ewt-sequences.json` beside it:

```python
import json
from pathlib import Path
from scipy.optimize import minimize
from sklearn.feature_extraction import DictVectorizer

rows = json.loads(Path("ewt-sequences.json").read_text(encoding="utf-8"))

def tag_index(tag):
    return 0 if tag in ("NOUN", "PROPN") else 1 if tag in ("VERB", "AUX") else 2

def features(words):
    return [{"bias": 1., "word": word.lower(), "suffix": word.lower()[-2:],
             "capital": float(word.istitle()),
             "previous": words[i-1].lower() if i else "<START>",
             "next": words[i+1].lower() if i+1 < len(words) else "<END>"}
            for i, word in enumerate(words)]

vectorizer = DictVectorizer(sparse=False)
vectorizer.fit([f for row in rows if row["split"] == "train"
                for f in features(row["tokens"])])
groups = {split: [(vectorizer.transform(features(row["tokens"])),
                   np.array([tag_index(t) for t in row["upos"]]))
                  for row in rows if row["split"] == split]
          for split in ("train", "dev", "test")}
feature_count = len(vectorizer.feature_names_)

def fit_model(structured):
    def objective(theta):
        weights = theta[:feature_count*3].reshape(feature_count, 3)
        transitions = (theta[feature_count*3:].reshape(3, 3)
                       if structured else np.zeros((3, 3)))
        loss = 0.; grad_w = np.zeros_like(weights); grad_a = np.zeros((3, 3))
        for x, gold in groups["train"]:
            emissions = x @ weights
            logz, nodes, edges, _ = infer(emissions, transitions)
            loss += (logz - emissions[np.arange(len(gold)), gold].sum()
                     - transitions[gold[:-1], gold[1:]].sum())
            residual = nodes.copy()
            residual[np.arange(len(gold)), gold] -= 1
            grad_w += x.T @ residual
            if structured:
                grad_a += edges.sum(axis=0)
                np.add.at(grad_a, (gold[:-1], gold[1:]), -1.)
        gradient = (np.r_[grad_w.ravel(), grad_a.ravel()]
                    if structured else grad_w.ravel())
        count = len(groups["train"])
        return loss/count + .05*np.dot(theta, theta), gradient/count + .1*theta
    initial = np.zeros(feature_count*3 + (9 if structured else 0))
    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      options={"maxiter": 150, "ftol": 1e-11, "gtol": 1e-7})
    weights = result.x[:feature_count*3].reshape(feature_count, 3)
    transitions = (result.x[feature_count*3:].reshape(3, 3)
                   if structured else np.zeros((3, 3)))
    return weights, transitions, result

def evaluate(model, split):
    weights, transitions, _ = model
    correct = total = whole = 0
    for x, gold in groups[split]:
        _, _, _, path = infer(x @ weights, transitions)
        correct += (gold == path).sum()
        total += len(gold)
        whole += np.array_equal(gold, path)
    return float(correct/total), float(whole/len(groups[split]))

models = {"independent": fit_model(False), "chain": fit_model(True)}
for name, model in models.items():
    print(name, model[2].success, np.round(evaluate(model, "dev"), 6))
selected = max(models, key=lambda name: evaluate(models[name], "dev")[0])
print("selected", selected, np.round(evaluate(models[selected], "test"), 6))
```

This implements the training mechanism with NumPy and SciPy's optimizer, rather than requiring a separate CRF package. `x` has shape `(sentence_length, feature_count)`; `weights` has shape `(feature_count, 3)`. Their product provides the input scores for the same inference routine used in the hand example. `grad_w` accumulates node-count errors; `grad_a` accumulates pair-count errors. There are 1,819 training-derived feature columns, 1,188 training tokens, 341 development tokens and 370 test tokens.

The bounded author calculation gave:

```text
independent True [0.788856 0.275   ]
chain True [0.824047 0.3     ]
selected chain [0.791892 0.175   ]
```

The first metric is token accuracy; the second is exact sentence accuracy. Development accuracy increased from 269/341 to 281/341. The chosen chain tagged 293/370 final-test tokens correctly, while getting every token correct in only 7/40 sentences. This is a practical illustration of the two error definitions from §3: quite good token accuracy can coexist with many sentences containing at least one error.

The selected model's test confusion matrix has gold labels in rows and predicted labels in columns:

| Gold \ Predicted | NOUN | VERB | OTHER |
| --- | ---: | ---: | ---: |
| NOUN | 49 | 1 | 26 |
| VERB | 8 | 42 | 23 |
| OTHER | 17 | 2 | 202 |

Missing noun/verb tokens by assigning OTHER is a visible weakness. A next **development** experiment could test richer suffix/context features or a different regularization value. The final test is already used here; further selection requires a new valid evaluation protocol. The experiment provides a controlled observation for this small extraction, not a general model ranking.

**Inline figure — inspect a real error.** A sentence selector shows aligned token, gold tag, predicted tag and the three marginal probabilities. Select errors from development by default. Next to the strip, show the chosen neighboring transition contribution, with an explicit note that the full path score includes the rest of the sentence. The recorded marginal probabilities describe uncertainty under this fitted model.

For a library variant, the [sklearn-crfsuite tutorial](https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html) explains dictionary features, fitting and inspecting weights. Its `c1/c2` conventions are library-specific; copying our \(\lambda\) number would not establish equal objectives. The package was absent from the author's environment, so this lesson does not present invented CRFsuite output or claim that the displayed NumPy result came from that library.

## 8. Deeper branch: neural inputs, constraints and scale

Read this after the core mechanism. A neural encoder can replace the manually designed input features without changing the chain's normalization. For a batch of \(B\) sentences padded to \(T\) positions, an encoder produces a tensor of shape \((B,T,H)\); a linear layer maps it to scores of shape \((B,T,K)\). The CRF adds a \((K,K)\) transition matrix, then computes gold path scores and \(\log Z\) for each real sequence.

Training minimizes \(\log Z-s_{\rm gold}\); gradients flow both into transitions and through input scores into the encoder. At inference, Viterbi replaces the training loss. Padding masks must remove padded positions from both computations, and the lengths must describe contiguous real tokens. A bidirectional recurrent encoder also needs correct handling of padding so reverse states do not incorporate artificial suffix tokens.

The [official PyTorch Bi-LSTM CRF tutorial](https://docs.pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html) provides a complete runnable neural example after its score convention is introduced. Its transition indexing must be read carefully: libraries can use current-by-previous ordering instead of this lesson's previous-by-current ordering. The neural tutorial is an optional implementation route, not a prerequisite for the CPU program above. The Lample et al. [NER paper, §2](https://aclanthology.org/N16-1030.pdf) describes character-level bidirectional LSTM representations and a chain output layer; a CRF's purpose remains scoring a structured output, whichever encoder supplies its inputs.

**Inline figure — one loss, two trainable parts.** Show token IDs → encoder `(B,T,H)` → input scores `(B,T,K)`, with transition scores entering the chain from a separate branch. Split the output into training's gold score/partition difference and inference's backpointer path. Draw the gradient returning along both trainable branches; do not imply that a discrete decoded path is differentiated during likelihood training.

For dense first-order transitions and fixed input scores, forward–backward and Viterbi require \(O(TK^2)\) arithmetic. A forward normalizer alone can keep \(O(K)\) working state; retained messages, backpointers or marginals need additional storage. Input feature computation or a neural encoder may dominate the total cost. Thus linear chain inference in \(T\) is not a measured throughput claim for an entire text system.

Larger label inventories increase the pair work quadratically. Sparse legal transitions can reduce work if the implementation exploits them. Higher-order label interactions remember more predecessor labels, expanding the state. A second-order chain can be represented using label pairs as states; its dense transitions require \(O(TK^3)\) rather than \(O(TK^2)\). These consequences follow from which indices must be enumerated, not from any preferred library.

## 9. Deeper branch: beyond a simple chain

A CRF is a conditional random field over output variables, not necessarily a chain. Its factors may join a group of pixels, repeated mentions in a document, or a whole segment. The general form is
\(p(y\mid x)\propto\prod_a\psi_a(y_a,x)\), where factor \(a\) involves the subset \(y_a\).

The factorization determines inference difficulty. Trees admit exact sum-product message passing. Adding long-range edges can produce loops; the simple left-to-right recurrence no longer applies. Exact elimination can build factors exponentially large in the graph's treewidth. Loopy belief propagation, variational methods and sampling then trade exactness for tractability. They need convergence and approximation diagnostics appropriate to the chosen method.

A **semi-Markov CRF** scores labeled segments rather than only individual adjacent labels. An address parser, for example, could score an entire multiword street-name span using its length and words. If segment length is capped at \(L\), a dense recurrence considers candidate segment starts and previous labels, typically \(O(TLK^2)\) plus segment feature computation. The segment representation changes the questions the model can ask; it is not merely a faster BIO decoder.

With a latent label subset \(h\), observed-label likelihood involves \(\log\sum_h e^{s(x,y,h)}-\log Z(x)\). That is a difference of log-sum-exp terms and is generally nonconcave. Partially labeled sequences can similarly sum over completions compatible with known labels. Fully unlabeled inputs alone contribute \(\log\sum_y p(y\mid x)=0\) to ordinary conditional likelihood, so using them requires an additional modeling or regularization assumption. The later semi-supervised lesson develops that missing bridge.

A fitted CRF can also **sample outputs conditional on a supplied input**. For a chain, first sample the final label from \(p(y_T\mid x)\); then work backward with
\[
p(y_t=i\mid y_{t+1}=j,x)\propto\alpha_t(i)e^{A_{ij}}.
\]
The next position's input contribution and suffix mass are constant across candidate \(i\) and cancel. This provides diverse plausible labelings for the same observation. A CRF alone has not modeled \(p(x)\), so it does not thereby generate new observations or score how common an input is.

A structured SVM keeps a compatible score-and-structure view but trains with a margin-based objective and loss-augmented decoding rather than a likelihood normalizer. A Bayesian CRF instead integrates over uncertain parameter weights when predicting. These are different extensions of the same basic modeling problem; neither is needed to complete the first-pass outcomes. The [canonical tutorial's related-work and frontier sections](https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf) provide further routes.

## 10. Practice: calculate, diagnose, transfer

Try each task before opening its hint or solution.

### 1. A changed path distribution

The input factors are `[2,3]` then `[2,2]`; pair factors are `[[2,1],[1,2]]`. Calculate the four masses, partition function, best path, and probability that the first label is A. Then decide whether separate marginal modes agree with the best path.

<details><summary>Hint</summary>

Multiply one entry from each input row and the selected pair entry. Keep the mass table in the order AA, AB, BA, BB.

</details>
<details><summary>Solution</summary>

The masses are `[8,4,6,12]`, totaling 30. `BB` wins with probability .4. The first A probability is `(8+4)/30=.4`; the second A probability is `(8+6)/30=7/15`. Both marginal modes are B, so they agree here. Agreement on this input does not replace the contrasting decision-loss example in §3.

</details>

### 2. What would one training update do?

Use practice 1's model, gold path `BA`, and an adjustable `B → A` log weight. Compute the log-likelihood derivative. Then include L2 regularization with \(\lambda=.2\) when that weight is currently zero.

<details><summary>Hint</summary>

The model's expected count is the probability of `BA`. One gold path contains either zero or one such edge.

</details>
<details><summary>Solution</summary>

The edge's model expectation is `6/30=.2`; the observed count is 1. The ascent derivative is `.8`. L2 subtracts \(\lambda w=0\), so the derivative remains `.8` at that point. After a positive update, regularization opposes further growth. Increasing this edge's weight by \(\log 2\) doubles `BA` mass from 6 to 12, giving partition 36 and probability `12/36=1/3`; the other masses remain 8,4,12.

</details>

### 3. Repair a probability calculation

A program obtains best masses `[8,12]` for paths ending at A and B in practice 1. It adds them and declares `Z=20`. Explain the error and compute the correct forward masses.

<details><summary>Hint</summary>

A forward cell includes every prefix ending at its label. List the two paths ending in each label.

</details>
<details><summary>Solution</summary>

The best-mass cells are Viterbi quantities. Forward masses are `AA+BA=14` and `AB+BB=16`; their sum is 30. Summing maxima loses the `BA` and `AB` alternatives. The Viterbi table itself is useful, but it cannot supply a likelihood normalizer.

</details>

### 4. Fix a tagging system's information contract

A system predicts entity spans as text arrives. Its training extractor uses the next word, the previous token's gold entity label, and `all_possible_transitions=False` to “ensure valid BIO.” Identify the three issues and propose a coherent repair.

<details><summary>Hint</summary>

Separate observed inputs, output variables, and library feature-generation settings.

</details>
<details><summary>Solution</summary>

The next word requires a stated delay or a causal extractor. The previous gold label belongs inside the structured target model, not as a training-only input; if using a predicted-label feature in another architecture, training must match that inference procedure. The library option does not define legal BIO edges. Use a decoder and normalizer with an explicit legal-path mask, including start rules, and test legal/illegal examples. These changes repair separate information, training and constraint problems.

</details>

### 5. An independent real-data investigation

On the supplied development split, change the feature extractor to omit previous and next words while retaining the chain. Keep the training set, label mapping, objective and regularization unchanged. Before fitting, record whether you expect noun recall, verb recall or neither to fall most, with a reason. Compare token accuracy, exact sentence accuracy and the confusion matrix. Inspect three changed predictions using sentence IDs. Do not use the final test to select the extractor.

<details><summary>Hint</summary>

Derive recall from a gold-label row: correct count divided by all items in that row. The original chain development noun recall is `52/72`; verb recall is `31/53`. Your new vocabulary must be fitted on training features again.

</details>
<details><summary>Solution and assessment</summary>

This is a new experiment, so there is no fabricated expected ranking. A complete answer supplies the saved prediction, actual training settings/stopping status, new development counts, three aligned before/after token examples, and a conclusion tied to those observations. Reproducing the original baseline first should give 281/341 token accuracy and 12/40 exact sentences, within normal numerical stability of the deterministic fit. A plausible hypothesis is that surrounding function words help identify noun/verb roles; if the aggregate result does not support it, retain that outcome and use the error slices to explain why. A higher score alone is insufficient without an unchanged evaluation protocol.

</details>

For the core finish line, explain how scores become probabilities, compute a changed two-label example, distinguish sums/maxima/marginals, derive one feature-count update, and diagnose unavailable features or a fake constraint. For the deeper route, additionally explain why a neural encoder changes optimization, why loops change inference, and how to sample labels conditional on a fixed input.

The next module entry is [Gaussian Processes](/learn/path/full-curriculum/gaussian-processes-gp?module=classical-ml). Both lessons use a structured probability model, but its objects change: a CRF connects discrete output labels, while a GP relates uncertain function values through a covariance kernel. Carry forward the habit of identifying the random variables, the conditioning information and the operation that makes their probabilities usable.

## References & another way to learn it

- [Sutton & McCallum — An Introduction to Conditional Random Fields](https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf). Canonical free tutorial. Read modeling §§2.3–2.5 after the core, inference §4 for the recurrences, estimation §5 for training, and §6.1.3 for label bias. A deeper text, with implementation and approximation detail beyond this lesson.
- [Official PyTorch — Advanced: Making Dynamic Decisions and the Bi-LSTM CRF](https://docs.pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html). A complete coding alternative after §8. The explanatory page and code were reviewed; its neural program was not run for this content checkpoint. Pay attention to its transition orientation and small demonstration corpus.
- [sklearn-crfsuite — Tutorial](https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html) and [API](https://sklearn-crfsuite.readthedocs.io/en/latest/api.html). Dictionary-feature workflow and weight inspection for readers who want a library fit. Read the API's parameter meanings rather than treating its settings as identical to the objective in §7. The documentation was inspected; package execution is deferred.
- [Lample et al. — Neural Architectures for Named Entity Recognition](https://aclanthology.org/N16-1030.pdf). Original research for the neural encoder/CRF connection; §2 describes the architecture. Useful after the neural branch, with recurrent-network prerequisites. Its reported research results concern the paper's own data and settings.
- [Universal Dependencies — English EWT](https://universaldependencies.org/treebanks/en_ewt/index.html) and [the r2.16 README](https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.16/README.md). Dataset description, annotation conventions and attribution for the real-text example; preserve the release and license when reusing the extract.
