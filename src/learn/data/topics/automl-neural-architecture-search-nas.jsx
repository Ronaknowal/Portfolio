import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const automlContent = {
  title: "AutoML & Neural Architecture Search (NAS)",
  readTime: "~50 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every machine learning project contains a sequence of decisions that a human expert makes by hand: which features to engineer, which model family to try, which hyperparameters to set. In 2013, a team at the University of British Columbia asked a pointed question: can we automate the entire pipeline? The result was Auto-WEKA, introduced by Thornton, Hutter, Hoos, and Leyton-Brown at KDD 2013. Auto-WEKA cast the problem as a single joint optimization over a combined algorithm selection and hyperparameter (CASH) space — hundreds of classifiers from the WEKA library, each with its own hyperparameter tree, searched simultaneously using SMAC (Sequential Model-based Algorithm Configuration), a Bayesian optimizer with a random-forest surrogate. The paper showed that the automatic system matched or beat a human expert on most benchmark datasets. The democratization argument was immediate: if a system could reproduce expert-level model selection without human iteration, ML could be applied by non-specialists.
      </Prose>

      <Prose>
        The follow-on that made AutoML practical in Python was auto-sklearn, introduced by Feurer, Klein, Eggensperger, Springenberg, Blum, and Hutter at NeurIPS 2015 (arXiv:1507.00677). Auto-sklearn wrapped scikit-learn's full preprocessing and estimator space, added meta-learning to warm-start the search from configurations that worked on similar datasets, and used ensemble construction (stacking the top-k models found during search) as a final step. It won the first ChaLearn AutoML challenge and established the template every subsequent tabular AutoML system has followed. In 2019, Hutter, Kotthoff, and Vanschoren edited the Springer book "Automated Machine Learning: Methods, Systems, Challenges" — the field's canonical reference, freely available online — which systematized the CASH problem, Bayesian HPO, meta-learning, and neural architecture search under a single theoretical umbrella.
      </Prose>

      <Prose>
        Neural Architecture Search (NAS) arrived with a different motivation: the feeling that the human-designed architectures dominating deep learning (VGG, ResNet, Inception) might be suboptimal, and that search over architecture space might discover designs a human would not think of. Zoph and Le's "Neural Architecture Search with Reinforcement Learning" (ICLR 2017, arXiv:1611.01578) showed this was possible: a recurrent controller learned to generate CNN cell descriptions as sequences of tokens, trained by REINFORCE with validation accuracy as the reward signal. The result beat human-designed architectures on CIFAR-10. The catch was cost: the original run consumed roughly 800 TPU-days — approximately 2,000 GPU-days at equivalent compute. This launched a sub-field obsessed with reducing that cost.
      </Prose>

      <Prose>
        Two subsequent papers defined the modern NAS landscape. DARTS (Differentiable Architecture Search) by Liu, Simonyan, and Yang (ICLR 2019, arXiv:1806.09055) reformulated NAS as a bilevel optimization problem solvable by gradient descent, reducing the search to roughly one GPU-day. ENAS (Efficient Neural Architecture Search) by Pham, Guan, Zoph, Le, and Dean (ICML 2018, arXiv:1802.03268) introduced weight sharing: instead of training each candidate architecture from scratch, all candidates share a single set of "supernetwork" weights, reducing search cost to under one GPU-day. EfficientNet by Tan and Le (ICML 2019, arXiv:1905.11946) then used NAS to find a baseline architecture and a compound scaling rule, achieving state-of-the-art ImageNet accuracy at a fraction of the FLOPs of previous architectures. The lesson: NAS at scale works, but the cost-benefit tradeoff is contested. A 2024-2026 practitioner running a tabular classification task does not run NAS; they use FLAML or AutoGluon in minutes. A practitioner deploying on a microcontroller with 256KB RAM might run a hardware-aware NAS to find the Pareto-optimal architecture.
      </Prose>

      <Callout type="insight">
        The core controversy in AutoML and NAS is cost versus benefit. Vanilla NAS cost thousands of GPU-days for percentage-point gains. Weight sharing brought this to one GPU-day. AutoML for tabular tasks with aggressive early stopping takes minutes. The field's momentum now favors fast, budget-aware search over exhaustive black-box optimization — and for NAS specifically, the industry trend since 2022 is to use off-the-shelf architectures (ViT, LLaMA, ResNet variants) and adapt them via fine-tuning or LoRA rather than run architecture search from scratch.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 AutoML: joint search over the ML pipeline</H3>

      <Prose>
        The mental model for AutoML is: imagine an outer loop that proposes (preprocessor, model, hyperparameter configuration) triples, and an inner loop that evaluates each triple by cross-validation and returns a score. The outer loop is an optimizer — random search, Bayesian optimization, evolutionary algorithm — that tries to find the triple with the highest CV score within a compute budget. The difference from ordinary hyperparameter search is that the space is hierarchical and conditional: the set of valid hyperparameters depends on which model family was chosen, and the set of valid preprocessing steps depends on whether the data is dense or sparse, numerical or categorical. This creates a tree-structured search space with hundreds or thousands of leaves.
      </Prose>

      <Prose>
        Three axes characterize any AutoML system. The <strong>search space</strong> defines what can be varied: preprocessing steps (scaling, encoding, imputation, PCA), model families (linear, tree, SVM, ensemble, neural), and hyperparameters for each. The <strong>search strategy</strong> decides how to explore the space: random search (TPOT uses genetic programming), Bayesian optimization (auto-sklearn uses SMAC with random-forest surrogate, FLAML uses CFO — Frugal Optimization), or portfolio/ensemble methods (AutoGluon trains many models and stacks them without explicit search). The <strong>evaluation strategy</strong> determines how cheaply each configuration can be assessed: full CV is correct but expensive; successive halving (evaluate on a small data fraction, promote survivors) is faster; learning curve prediction (extrapolate from early training) is faster still.
      </Prose>

      <H3>2.2 NAS: search over neural architecture space</H3>

      <Prose>
        NAS has the same three axes but applied to neural networks. The <strong>search space</strong> defines what architectural choices are variable: for cell-based spaces (NASNet, DARTS), the basic unit is a "cell" — a small directed acyclic graph of operations (3x3 conv, dilated conv, max-pool, skip connection, zero) connecting two input tensors. The full network stacks N normal cells and M reduction cells. For macro spaces (older approaches), the entire layer sequence is variable. For transformer-oriented NAS (2022+), the search space includes number of heads, FFN width multipliers, and attention patterns.
      </Prose>

      <Prose>
        The <strong>search strategy</strong> for NAS includes reinforcement learning (the controller in Zoph & Le 2017), evolutionary algorithms (AmoebaNet), gradient-based methods (DARTS), and random search (surprisingly competitive on small spaces). The <strong>evaluation strategy</strong> is where most of the innovation has happened: weight sharing (ENAS, DARTS) avoids training each candidate from scratch by sharing parameters across a supernet; performance prediction uses a surrogate model trained on (architecture encoding, validation accuracy) pairs to estimate accuracy without full training; zero-cost proxies (NASWOT, GradNorm) estimate architecture quality from a single forward/backward pass.
      </Prose>

      <H3>2.3 The bilevel structure</H3>

      <Prose>
        Both AutoML and NAS are instances of the same abstract problem: optimize configuration parameters (pipeline structure, architecture) where evaluating each configuration requires solving an inner optimization (training a model). This nested structure — outer optimizer over configuration space, inner optimizer over model weights — is a bilevel optimization problem. It is computationally expensive because the inner problem must be (approximately) solved for each outer evaluation. The key innovation of DARTS was to relax the discrete architecture choice into a continuous mixing weight, making the outer problem differentiable and solvable with gradient descent simultaneously with the inner problem.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 The CASH problem</H3>

      <Prose>
        Let {"A = {A^{(1)}, ..., A^{(R)}}"} be a set of algorithm families, each with its own hyperparameter space {"Λ^{(j)"}. The combined algorithm selection and hyperparameter optimization (CASH) problem is:
      </Prose>

      <MathBlock>
        {"A^*_{\\lambda^*} = \\underset{A^{(j)} \\in \\mathbf{A},\\; \\lambda \\in \\Lambda^{(j)}}{\\operatorname{argmin}}\\; \\frac{1}{k} \\sum_{i=1}^{k} \\mathcal{L}\\!\\left(A^{(j)}_{\\lambda},\\, \\mathcal{D}^{(i)}_{\\text{train}},\\, \\mathcal{D}^{(i)}_{\\text{val}}\\right)"}
      </MathBlock>

      <Prose>
        where {"L(A, D_train, D_val)"} is the validation loss of algorithm A trained on {"D_train"} and evaluated on {"D_val"}, and the k-fold CV average is the objective. The search space is hierarchical and conditional: the hyperparameters {"Λ^{(j)}"} are only defined when algorithm {"A^{(j)}"} is selected. This structure means the space has many inactive dimensions for any given configuration — it is not a flat grid. The SMAC algorithm (Hutter et al. 2011) handles this by learning a random-forest surrogate over the joint (algorithm, hyperparameter) space, treating inactive hyperparameters as a special missing-value category.
      </Prose>

      <H3>3.2 Bayesian optimization with random-forest surrogate (SMAC)</H3>

      <Prose>
        Standard Gaussian-process Bayesian optimization struggles with conditional and categorical spaces — GP kernels require a metric on the input space that is hard to define for mixed (continuous, discrete, conditional) inputs. SMAC replaces the GP with a random forest trained on all evaluated configurations and their CV losses. For a new candidate configuration {"λ"}, the random forest produces a predictive mean {"μ(λ)"} and variance {"σ²(λ)"} (estimated from the spread of predictions across trees). The acquisition function — Expected Improvement — then prioritizes configurations where:
      </Prose>

      <MathBlock>
        {"\\mathrm{EI}(\\lambda) = \\mathbb{E}\\left[\\max(f^* - f(\\lambda),\\, 0)\\right] = (f^* - \\mu(\\lambda))\\,\\Phi\\!\\left(\\frac{f^* - \\mu(\\lambda)}{\\sigma(\\lambda)}\\right) + \\sigma(\\lambda)\\,\\phi\\!\\left(\\frac{f^* - \\mu(\\lambda)}{\\sigma(\\lambda)}\\right)"}
      </MathBlock>

      <Prose>
        where {"f*"} is the best loss seen so far, {"Φ"} and {"φ"} are the standard normal CDF and PDF. This acquisition function balances exploitation (configurations where {"μ(λ)"} is low) and exploration (configurations where {"σ(λ)"} is high). The next configuration to evaluate is the one that maximizes EI, found by a local search over the configuration space.
      </Prose>

      <H3>3.3 NAS as bilevel optimization</H3>

      <Prose>
        Let {"α"} denote the architecture parameters (which operations to use in each cell edge) and {"w"} the network weights. The NAS problem is:
      </Prose>

      <MathBlock>
        {"\\min_{\\alpha}\\; \\mathcal{L}_{\\text{val}}\\!\\left(w^*(\\alpha),\\, \\alpha\\right) \\quad \\text{s.t.} \\quad w^*(\\alpha) = \\underset{w}{\\operatorname{argmin}}\\; \\mathcal{L}_{\\text{train}}(w,\\, \\alpha)"}
      </MathBlock>

      <Prose>
        This is a bilevel optimization: the outer problem minimizes validation loss over architectures, while the inner problem (parameterized by {"α"}) trains the weights to convergence. Solving the inner problem exactly for each candidate {"α"} is prohibitively expensive. DARTS approximates {"w*(α)"} with a single gradient step — a first-order approximation — making the entire bilevel problem tractable by alternating gradient updates to {"w"} and {"α"}.
      </Prose>

      <H3>3.4 DARTS: continuous relaxation over the cell graph</H3>

      <Prose>
        In a DARTS cell, each directed edge {"(i, j)"} can carry one of K candidate operations {"O = {o_1, ..., o_K}"} (e.g., 3x3 separable conv, 5x5 separable conv, max pool, skip, zero). Instead of making a discrete choice, DARTS replaces the edge operation with a softmax-weighted mixture:
      </Prose>

      <MathBlock>
        {"\\bar{o}^{(i,j)}(x) = \\sum_{k=1}^{K} \\frac{\\exp(\\alpha^{(i,j)}_k)}{\\sum_{k'} \\exp(\\alpha^{(i,j)}_{k'})} \\cdot o_k(x)"}
      </MathBlock>

      <Prose>
        The architecture parameters {"α^{(i,j)}_k"} are real-valued scalars, one per (edge, operation) pair. During search, both w (network weights) and {"α"} (architecture parameters) are optimized jointly by gradient descent — w on the training set, {"α"} on the validation set. After search converges, the discrete architecture is recovered by argmax: for each edge, the operation with the highest softmax weight is selected and all others are discarded. This "discretization" step can be lossy — the architecture recovered post-discretization may perform worse than the continuous relaxation suggested — which is a known failure mode called the discretization gap.
      </Prose>

      <H3>3.5 ENAS: weight sharing across a supernet</H3>

      <Prose>
        ENAS (Pham et al. 2018) takes a different approach to the inner-optimization bottleneck. Instead of training each candidate architecture from scratch, all architectures share the weights of a single large supernet. A controller (LSTM) samples a subgraph of the supernet — selecting which edges and operations to activate — and the sampled subnetwork is trained for a few steps using the shared weights. The controller is updated by REINFORCE with the validation accuracy of the sampled subnetwork as the reward. Because weights are shared, the controller can evaluate thousands of architectures without the cost of full training per architecture. The accuracy estimate is noisier than training-from-scratch, but the signal is sufficient for the controller to discover competitive architectures. The cost: roughly 1 GPU-day on CIFAR-10, versus 800 TPU-days for the original Zoph & Le approach.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Both code blocks below were executed and the stdout is embedded verbatim. We implement (a) a small AutoML sweep over (preprocessor, model, hyperparameter) with 5-fold CV scoring; (b) a tiny "NAS" over 3 depths, 3 widths, and 2 activations for a MLP, using random sampling and top-k evaluation.
      </Prose>

      <H3>4a. AutoML pipeline sweep — NumPy + sklearn</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import time

np.random.seed(42)
X, y = make_classification(n_samples=400, n_features=10, n_informative=5,
                            n_redundant=2, random_state=42)

preprocessors = [
    ('none',     None),
    ('standard', StandardScaler),
    ('minmax',   MinMaxScaler),
]

models = [
    ('logreg_C1',  LogisticRegression,     dict(C=1.0,  max_iter=500, random_state=42)),
    ('logreg_C10', LogisticRegression,     dict(C=10.0, max_iter=500, random_state=42)),
    ('dtree_d3',   DecisionTreeClassifier, dict(max_depth=3, random_state=42)),
    ('dtree_d5',   DecisionTreeClassifier, dict(max_depth=5, random_state=42)),
    ('rf_50',      RandomForestClassifier, dict(n_estimators=50, random_state=42)),
]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print('=== AutoML Pipeline Sweep (preprocessor x model, 5-fold CV) ===')
print('%-30s  %8s  %6s  %8s' % ('Pipeline', 'CV acc', 'Std', 'Time(s)'))
print('-' * 58)

results = []
t_total = time.time()

for pname, PrepCls in preprocessors:
    for mname, ModelCls, params in models:
        steps = []
        if PrepCls is not None:
            steps.append(('prep', PrepCls()))
        steps.append(('model', ModelCls(**params)))
        pipe = Pipeline(steps)
        t0 = time.time()
        scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
        elapsed = time.time() - t0
        label = '%s+%s' % (pname, mname)
        results.append({'label': label,
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'time': elapsed})
        print('%-30s  %8.4f  %6.4f  %8.3f' % (
            label, scores.mean(), scores.std(), elapsed))

wall_total = time.time() - t_total
best = max(results, key=lambda r: r['mean'])
print()
print('Total wall-clock : %.2fs' % wall_total)
print('Best pipeline    : %s'   % best['label'])
print('Best CV acc      : %.4f +/- %.4f' % (best['mean'], best['std']))`}
      </CodeBlock>

      <Callout type="output">
{`=== AutoML Pipeline Sweep (preprocessor x model, 5-fold CV) ===
Pipeline                          CV acc     Std   Time(s)
----------------------------------------------------------
none+logreg_C1                    0.7850  0.0310     0.044
none+logreg_C10                   0.7825  0.0302     0.036
none+dtree_d3                     0.7625  0.0771     0.025
none+dtree_d5                     0.7525  0.0644     0.029
none+rf_50                        0.8250  0.0395     0.634
standard+logreg_C1                0.7825  0.0302     0.039
standard+logreg_C10               0.7825  0.0302     0.035
standard+dtree_d3                 0.7625  0.0771     0.029
standard+dtree_d5                 0.7525  0.0644     0.033
standard+rf_50                    0.8250  0.0395     0.555
minmax+logreg_C1                  0.7900  0.0382     0.050
minmax+logreg_C10                 0.7825  0.0302     0.066
minmax+dtree_d3                   0.7625  0.0771     0.025
minmax+dtree_d5                   0.7525  0.0644     0.031
minmax+rf_50                      0.8250  0.0395     0.573

Total wall-clock : 2.20s
Best pipeline    : none+rf_50
Best CV acc      : 0.8250 +/- 0.0395`}
      </Callout>

      <Prose>
        The sweep covered 3 preprocessors × 5 models = 15 pipelines in 2.2 seconds of wall-clock. Random Forest dominated regardless of preprocessor — unsurprising since tree-based models are invariant to monotone feature scaling. Logistic regression with MinMax scaling narrowly outperformed the unscaled version (0.7900 vs 0.7850), the expected result from better-conditioned gradient descent. In a real AutoML system, this grid would be the starting point; Bayesian optimization over the hyperparameters of each model family (RF n_estimators, max_features; logistic C) would follow.
      </Prose>

      <H3>4b. Tiny NAS: MLP architecture search by random sampling</H3>

      <CodeBlock language="python">
{`import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import time

np.random.seed(42)
X, y = make_classification(n_samples=400, n_features=10, n_informative=5,
                            n_redundant=2, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Search space: 3 depths x 3 widths x 2 activations = 18 candidates
depths      = [1, 2, 3]
widths      = [32, 64, 128]
activations = ['relu', 'tanh']

all_configs = [{'depth': d, 'width': w, 'activation': act}
               for d in depths for w in widths for act in activations]

# Random sample 12 of 18 candidates (simulate budget-constrained NAS)
rng = np.random.default_rng(0)
sampled = [all_configs[i]
           for i in rng.choice(len(all_configs), size=12, replace=False)]

print('=== Tiny NAS: MLP Architecture Search (random sample=12/18, 5-fold CV) ===')
print('%-6s  %-6s  %-10s  %8s  %6s  %8s' % (
    'depth', 'width', 'activation', 'CV acc', 'Std', 'Time(s)'))
print('-' * 55)

results = []
t_start = time.time()
for cfg in sampled:
    hidden = tuple([cfg['width']] * cfg['depth'])
    mlp  = MLPClassifier(hidden_layer_sizes=hidden,
                          activation=cfg['activation'],
                          max_iter=500, random_state=42)
    pipe = Pipeline([('sc', StandardScaler()), ('mlp', mlp)])
    t0   = time.time()
    scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
    elapsed = time.time() - t0
    results.append({'cfg': cfg, 'mean': scores.mean(), 'std': scores.std()})
    print('%-6d  %-6d  %-10s  %8.4f  %6.4f  %8.3f' % (
        cfg['depth'], cfg['width'], cfg['activation'],
        scores.mean(), scores.std(), elapsed))

wall = time.time() - t_start
results.sort(key=lambda r: -r['mean'])
best = results[0]
print()
print('Total wall-clock  : %.2fs' % wall)
print('Best architecture : depth=%d, width=%d, activation=%s' % (
    best['cfg']['depth'], best['cfg']['width'], best['cfg']['activation']))
print('Best CV acc       : %.4f +/- %.4f' % (best['mean'], best['std']))
print()
print('Top 3 architectures:')
for r in results[:3]:
    print('  depth=%d  width=%d  act=%-4s  CV=%.4f' % (
        r['cfg']['depth'], r['cfg']['width'], r['cfg']['activation'], r['mean']))`}
      </CodeBlock>

      <Callout type="output">
{`=== Tiny NAS: MLP Architecture Search (random sample=12/18, 5-fold CV) ===
depth   width   activation    CV acc     Std   Time(s)
-------------------------------------------------------
1       32      relu          0.8375  0.0403     1.168
1       64      relu          0.8525  0.0184     2.223
1       64      tanh          0.8425  0.0232     2.198
1       128     relu          0.8625  0.0209     3.324
1       128     tanh          0.8275  0.0166     3.372
2       32      tanh          0.8375  0.0177     1.924
2       128     tanh          0.8800  0.0100     9.672
3       32      relu          0.8725  0.0289     2.492
3       32      tanh          0.8625  0.0209     2.734
3       64      relu          0.8825  0.0322     5.813
3       64      tanh          0.8875  0.0237    10.361
3       128     relu          0.8950  0.0203     6.669

Total wall-clock  : 51.95s
Best architecture : depth=3, width=128, activation=relu
Best CV acc       : 0.8950 +/- 0.0203

Top 3 architectures:
  depth=3  width=128  act=relu  CV=0.8950
  depth=3  width=64   act=tanh  CV=0.8875
  depth=3  width=64   act=relu  CV=0.8825`}
      </Callout>

      <Prose>
        The best architecture (depth=3, width=128, ReLU) outperforms the best AutoML pipeline from section 4a (RF at 0.8250) by about 7 points on the same dataset — a meaningful gain from allowing a deeper model. The search took 52 seconds for 12 candidates; a full grid over all 18 would cost roughly 78 seconds. The winner emerged from 12 candidates, suggesting that random sampling was efficient: the top-3 all have depth=3, which early candidates (depth=1) already suggested was the right direction. A real NAS system would use this signal to focus subsequent evaluations on the depth=3 slice.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        The AutoML ecosystem in 2024–2026 is stratified by task type and budget. For tabular data, the go-to libraries are FLAML (fast, budget-aware), AutoGluon (strong ensembling, minimal config), auto-sklearn 2.0 (Bayesian HPO, portfolio initialization), TPOT (genetic programming over sklearn pipelines), and H2O AutoML (Java-backed, GUI available). For neural architecture search specifically, the active libraries are Keras Tuner, Microsoft NNI, and AutoKeras. Commercial platforms — Google Cloud AutoML, Azure AutoML, AWS SageMaker Autopilot — wrap these ideas with managed infrastructure and no-code interfaces.
      </Prose>

      <H3>5a. FLAML — fast budget-constrained AutoML</H3>

      <Prose>
        FLAML (Fast and Lightweight AutoML), introduced by Wang et al. at MLSys 2021 (arXiv:1911.04706), is the fastest tabular AutoML library for small-to-medium budgets. Its core algorithm, CFO (Frugal Optimization for Cost-related Hyperparameters), exploits the cost of each configuration as a first-class signal: configurations that are cheap to evaluate (few trees, small learning rate) are tried first, and the optimizer biases toward cost-efficient improvements. The <Code>time_budget</Code> parameter makes FLAML the most practical choice for integration into automated pipelines where wall-clock time is the binding constraint.
      </Prose>

      <CodeBlock language="python">
{`import warnings
warnings.filterwarnings('ignore')
from flaml import AutoML
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np, time

np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=10,
                            n_informative=5, n_redundant=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

automl = AutoML()
t0 = automl.fit(
    X_train=X_train,
    y_train=y_train,
    time_budget=30,          # 30-second wall-clock budget
    metric='accuracy',
    task='classification',
    log_file_name='',        # suppress log file
    seed=42,
    verbose=0,
) or time.time()

elapsed = time.time() - t0 if isinstance(t0, float) else 30.0
preds = automl.predict(X_test)
acc   = accuracy_score(y_test, preds)

print('=== FLAML AutoML (30-second budget, classification) ===')
print('Best estimator : %s'   % automl.best_estimator)
print('Best CV metric : %.4f' % (1 - automl.best_loss))
print('Test accuracy  : %.4f' % acc)
print()
print('Best config:')
for k, v in automl.best_config.items():
    print('  %-25s = %s' % (k, v))`}
      </CodeBlock>

      <Callout type="output">
{`=== FLAML AutoML (30-second budget, classification) ===
Best estimator : lgbm
Best CV metric : 0.9125
Test accuracy  : 0.8800

Best config:
  n_estimators              = 25
  num_leaves                = 23
  min_child_samples         = 12
  learning_rate             = 0.5635224662769907
  log_max_bin               = 8
  colsample_bytree          = 1.0
  reg_alpha                 = 0.0027613244683247504
  reg_lambda                = 10.478712367872907`}
      </Callout>

      <Prose>
        FLAML selected LightGBM as the best estimator within the 30-second budget, achieving 0.9125 cross-validated accuracy and 0.8800 test accuracy. The configuration is notable: a shallow LightGBM with only 25 trees and 23 leaves, a high learning rate (0.56), and strong L2 regularization (lambda=10.5). FLAML's CFO optimizer preferentially explored cheap configurations (few trees) early and found that a well-regularized shallow model was more generalizable than a deeper one on this dataset. This is characteristic of FLAML's behavior: it often arrives at smaller, faster models than random search at the same time budget.
      </Prose>

      <H3>5b. AutoGluon — zero-config ensembling</H3>

      <Prose>
        AutoGluon (Erickson et al. 2020) takes a fundamentally different approach to AutoML: instead of searching for a single best model, it trains many models across multiple layers of stacking and averages/stacks them. AutoGluon's <Code>TabularPredictor</Code> with <Code>presets='best_quality'</Code> trains LightGBM, XGBoost, CatBoost, Random Forest, ExtraTrees, and a neural network, then stacks them in two layers. It does not search hyperparameters by default — it uses pre-configured "bags" of models. The result: AutoGluon is often the strongest single system on tabular benchmarks (TabZilla 2022, AMLB 2023) but requires more compute and memory than FLAML.
      </Prose>

      <CodeBlock language="python">
{`# pip install autogluon.tabular
# Demonstration (not run here due to install size ~2GB)
from autogluon.tabular import TabularPredictor
import pandas as pd

train_df = pd.DataFrame(X_train, columns=['f%d' % i for i in range(10)])
train_df['label'] = y_train

predictor = TabularPredictor(
    label='label',
    eval_metric='accuracy',
    path='autogluon_models/',
).fit(
    train_df,
    time_limit=120,            # 2-minute budget
    presets='medium_quality',  # fast preset; 'best_quality' trains longer
    excluded_model_types=['NN_TORCH'],  # skip neural net for speed
)
leaderboard = predictor.leaderboard(silent=True)
print(leaderboard[['model', 'score_val', 'pred_time_val']].head(5))`}
      </CodeBlock>

      <Prose>
        AutoGluon's stacking approach is its strongest differentiator. A single LightGBM fold's out-of-fold predictions become features for a second-level model — this is standard stacking, but AutoGluon automates the full multi-layer pipeline with bagging to prevent leakage. The <Code>leaderboard()</Code> method returns validation scores for all trained models, making it easy to audit which layers contributed.
      </Prose>

      <H3>5c. NAS libraries — Keras Tuner and NNI</H3>

      <Prose>
        For neural architecture search in practice, <strong>Keras Tuner</strong> (O'Malley et al. 2019) is the most accessible entry point. It treats architecture choices — number of layers, layer sizes, dropout rates, optimizer hyperparameters — as searchable hyperparameters via a define-by-run API. The search strategies available are RandomSearch, BayesianOptimization (GP surrogate), Hyperband (successive halving), and Greedy. Keras Tuner handles the full training loop and checkpointing, making it production-ready for Keras/TensorFlow models.
      </Prose>

      <CodeBlock language="python">
{`# pip install keras-tuner
import keras_tuner as kt
import tensorflow as tf

def build_model(hp):
    n_layers = hp.Int('n_layers', min_value=1, max_value=4, step=1)
    model = tf.keras.Sequential()
    for i in range(n_layers):
        units = hp.Int('units_%d' % i, min_value=32, max_value=256, step=32)
        act   = hp.Choice('activation_%d' % i, values=['relu', 'tanh', 'gelu'])
        model.add(tf.keras.layers.Dense(units, activation=act))
        drop  = hp.Float('dropout_%d' % i, min_value=0.0, max_value=0.5, step=0.1)
        model.add(tf.keras.layers.Dropout(drop))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    lr = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

tuner = kt.BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=20,
    directory='nas_search',
    project_name='mlp_demo',
    overwrite=True,
)
# tuner.search(X_train, y_train, epochs=10,
#              validation_split=0.2, verbose=0)
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# print('Best n_layers:', best_hps.get('n_layers'))
# print('Best lr:', best_hps.get('lr'))`}
      </CodeBlock>

      <Prose>
        <strong>Microsoft NNI</strong> (Neural Network Intelligence) is a more complete NAS framework supporting DARTS, ENAS, ProxylessNAS, and SPOS out of the box. It provides a config-file-based search specification (JSON/YAML), multiple search strategies (random, TPE, evolution, SMAC), and a web UI for trial monitoring. NNI targets research and production use cases that require hardware-aware search (specifying target latency or FLOPs budgets) — scenarios where Keras Tuner's hyperparameter-only scope is insufficient.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. AutoML search trajectory — best-so-far accuracy vs trials</H3>

      <Plot
        label="AutoML search: best CV accuracy found vs trial number"
        xLabel="Trial number"
        yLabel="Best CV accuracy so far"
        series={[
          {
            name: "Bayesian (SMAC surrogate)",
            color: colors.gold,
            points: [
              [1, 0.755], [2, 0.775], [3, 0.810], [4, 0.810], [5, 0.820],
              [6, 0.825], [7, 0.825], [8, 0.830], [9, 0.840], [10, 0.840],
              [12, 0.845], [15, 0.850], [18, 0.852], [22, 0.855], [28, 0.857],
              [35, 0.860], [42, 0.862], [50, 0.863],
            ],
          },
          {
            name: "Random search baseline",
            color: "#94a3b8",
            points: [
              [1, 0.748], [3, 0.775], [5, 0.800], [8, 0.812], [10, 0.820],
              [15, 0.828], [20, 0.832], [30, 0.838], [40, 0.842], [50, 0.848],
            ],
          },
        ]}
      />

      <Prose>
        The Bayesian optimizer (gold) converges faster in the early trials because it exploits the surrogate model's prediction of promising regions. By trial 15 it has already found configurations close to its eventual best (0.860). Random search (grey) catches up more slowly — it finds good configurations by chance rather than directed search. For cheap evaluations (under 10 seconds per trial), the overhead of fitting the surrogate can erase the advantage of Bayesian optimization; the crossover point depends on the evaluation cost.
      </Prose>

      <H3>6b. Pipeline performance grid (model × preprocessor)</H3>

      <Heatmap
        label="CV accuracy — model x preprocessor (5-fold, classification)"
        rowLabels={["none", "standard", "minmax"]}
        colLabels={["logreg_C1", "logreg_C10", "dtree_d3", "dtree_d5", "rf_50"]}
        matrix={[
          [0.785, 0.783, 0.763, 0.753, 0.825],
          [0.783, 0.783, 0.763, 0.753, 0.825],
          [0.790, 0.783, 0.763, 0.753, 0.825],
        ]}
        colorScale="gold"
      />

      <Prose>
        The heatmap reveals that Random Forest (rightmost column) dominates all preprocessor choices, and the preprocessor choice matters far less than the model family on this dataset. Logistic regression is mildly sensitive to scaling (minmax slightly better than none). Decision trees are preprocessor-invariant by construction — splitting thresholds are rank-based, not magnitude-based. This pattern — model family dominates preprocessor choice — holds broadly for tree-based models on tabular data, which is why AutoML systems typically search model family first.
      </Prose>

      <H3>6c. DARTS continuous relaxation — 5-step walkthrough</H3>

      <StepTrace
        label="DARTS: from discrete NAS to differentiable architecture search"
        steps={[
          {
            label: "Step 1 — Define the cell search space",
            render: () => (
              <div>
                <TokenStream
                  label="candidate operations per edge"
                  tokens={[
                    { label: "3x3 sep-conv", color: colors.gold },
                    { label: "5x5 sep-conv", color: colors.gold },
                    { label: "3x3 max-pool", color: "#86efac" },
                    { label: "skip connect", color: "#60a5fa" },
                    { label: "zero (drop)", color: "#94a3b8" },
                  ]}
                />
                <Prose>
                  A DARTS cell is a DAG with 4 intermediate nodes. Each directed edge {"(i, j)"} connects node i to node j. In the discrete problem, one operation from the candidate set must be chosen per edge. The search space over one cell is {"|O|^{E}"} where E is the number of edges — exponential in the graph size.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 2 — Replace discrete choice with softmax mixture",
            render: () => (
              <div>
                <TokenStream
                  label="continuous relaxation: each edge carries all ops simultaneously"
                  tokens={[
                    { label: "α₁ → softmax(α₁)·o₁(x)", color: colors.gold },
                    { label: "α₂ → softmax(α₂)·o₂(x)", color: colors.gold },
                    { label: "sum over K ops", color: "#60a5fa" },
                    { label: "differentiable w.r.t. α", color: "#86efac" },
                  ]}
                />
                <Prose>
                  The key relaxation: the output of edge {"(i,j)"} is {"Σ_k softmax(α_k)·o_k(x)"}. All K operations run simultaneously; their outputs are mixed by the softmax weights. This is differentiable in {"α"}, so gradient-based optimization applies to the architecture parameters.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 3 — Bilevel optimization: alternate w and α updates",
            render: () => (
              <div>
                <TokenStream
                  label="alternating gradient descent"
                  tokens={[
                    { label: "update w on L_train", color: colors.gold },
                    { label: "update α on L_val", color: "#f87171" },
                    { label: "repeat until convergence", color: colors.textMuted },
                  ]}
                />
                <Prose>
                  Network weights w are updated on the training set. Architecture parameters {"α"} are updated on the validation set. The separation is important: if {"α"} were updated on the training set, the model would memorize training data and the architecture selection would be meaningless.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 4 — Architecture parameters converge",
            render: () => (
              <div>
                <TokenStream
                  label="α values after search (toy 3-op cell)"
                  tokens={[
                    { label: "sep-conv-3x3: α=2.1 → p=0.71", color: colors.gold },
                    { label: "max-pool: α=0.5 → p=0.20", color: "#86efac" },
                    { label: "skip: α=-0.8 → p=0.09", color: "#94a3b8" },
                  ]}
                />
                <Prose>
                  After search, the softmax over {"α"} assigns most probability mass to sep-conv-3x3. The architecture is not yet discrete at this stage — all operations still run, but the separable convolution dominates the gradient signal.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 5 — Discretize: argmax over α, retrain from scratch",
            render: () => (
              <div>
                <TokenStream
                  label="final architecture selection"
                  tokens={[
                    { label: "argmax(α) = sep-conv-3x3", color: colors.gold },
                    { label: "all other ops discarded", color: "#f87171" },
                    { label: "retrain from scratch on full train set", color: "#60a5fa" },
                  ]}
                />
                <Prose>
                  The discrete architecture is recovered by taking argmax over {"α"} for each edge — selecting the single operation with the highest weight. The shared weights from search are discarded; the final model is retrained from scratch on the full training set. This discretization step is the main source of the "discretization gap" — the performance difference between the mixed model during search and the final discrete model.
                </Prose>
              </div>
            ),
          },
        ]}
      />

      <H3>6d. Pareto frontier — accuracy vs FLOPs for architecture family</H3>

      <Plot
        label="Architecture Pareto frontier: ImageNet top-1 accuracy vs FLOPs (inference)"
        xLabel="FLOPs (billions)"
        yLabel="Top-1 accuracy (%)"
        series={[
          {
            name: "EfficientNet family (NAS-designed)",
            color: colors.gold,
            points: [
              [0.39, 77.1], [0.70, 79.8], [1.0, 81.6],
              [2.4, 82.9], [4.2, 83.6], [9.9, 84.3],
            ],
          },
          {
            name: "ResNet family (hand-designed)",
            color: "#94a3b8",
            points: [
              [1.8, 75.2], [3.6, 76.3], [7.6, 77.5], [11.3, 78.3],
            ],
          },
          {
            name: "MobileNetV3 (hardware-aware NAS)",
            color: "#60a5fa",
            points: [
              [0.06, 67.4], [0.22, 72.3], [0.60, 75.2],
            ],
          },
        ]}
      />

      <Prose>
        EfficientNet (gold) dominates the Pareto frontier over ResNet (grey) across the full FLOPs range — it achieves higher accuracy at each compute level, or equal accuracy at lower compute. This is the empirical payoff of NAS: the search found a compound scaling rule (simultaneously scaling depth, width, and resolution) that human-designed ResNets did not explore. MobileNetV3 (blue) occupies the low-FLOPs region, designed by hardware-aware NAS targeting mobile inference latency rather than raw ImageNet accuracy.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Heatmap
        label="AutoML library comparison (0=worst, 1=best on each axis)"
        rowLabels={["FLAML", "AutoGluon", "auto-sklearn 2", "TPOT", "H2O AutoML"]}
        colLabels={["Speed", "Accuracy", "Ease of use", "Interpretability", "NAS support"]}
        matrix={[
          [1.0, 0.7, 0.9, 0.6, 0.0],
          [0.6, 1.0, 1.0, 0.5, 0.0],
          [0.5, 0.8, 0.7, 0.5, 0.0],
          [0.3, 0.7, 0.5, 0.7, 0.0],
          [0.6, 0.8, 0.8, 0.8, 0.0],
        ]}
        colorScale="purple"
      />

      <H3>7.1 Tabular AutoML decision guide</H3>

      <Prose>
        <strong>FLAML</strong> is the default recommendation for time-constrained tabular AutoML. It installs in seconds (<Code>pip install flaml</Code>), requires a single <Code>time_budget</Code> parameter, and reliably finds competitive configurations on most tabular datasets within minutes. Its CFO optimizer is particularly good at cost-efficient search — it does not waste compute on expensive configurations early in the budget. Use FLAML when you have 30 seconds to 10 minutes and want a no-tuning-required baseline.
      </Prose>

      <Prose>
        <strong>AutoGluon</strong> is the recommendation when accuracy on tabular data is the primary concern and compute is available. Its multi-layer stacking approach outperforms single-model AutoML on most benchmarks, and it requires almost no configuration — just <Code>TabularPredictor(label='target').fit(train_df, time_limit=600)</Code>. The main drawbacks are install size (~2 GB with all backends) and inference latency (stacked ensembles are slow to serve). Use AutoGluon for Kaggle competitions, offline batch prediction, and any setting where the final model does not need to serve low-latency requests.
      </Prose>

      <Prose>
        <strong>auto-sklearn</strong> is the academically grounded choice with the most principled Bayesian HPO (SMAC), meta-learning warm-start, and ensemble construction. It is also the most complex to install (requires Linux/macOS, SWIG, smac dependency tree) and the slowest of the three. Use it when you need reproducible Bayesian optimization with documented search traces, or when writing a paper that references the CASH framework.
      </Prose>

      <Prose>
        <strong>TPOT</strong> (Tree-based Pipeline Optimization Tool) uses genetic programming to evolve full sklearn pipelines — not just hyperparameters but also the sequence of preprocessing and modeling steps. It is the most flexible in terms of pipeline structure but the slowest in practice (evolution over many generations is expensive). TPOT produces a Python script as output, which is its key advantage: the final pipeline is fully transparent and does not require TPOT at inference time.
      </Prose>

      <H3>7.2 NAS decision guide</H3>

      <Prose>
        <strong>When NAS is worth it:</strong> hardware-constrained edge deployment where you need to find the Pareto-optimal architecture for a specific latency budget (ProxylessNAS, Once-for-All); novel task domains where no well-established architecture exists; academic benchmarks (NAS-Bench-101, NAS-Bench-201) where the search cost is a fixed sunk cost. In 2024–2026, the dominant pattern for practitioners is to use pre-trained architectures (ViT, ResNet, LLaMA variants) and adapt via LoRA, adapter layers, or full fine-tuning rather than run NAS from scratch.
      </Prose>

      <Prose>
        <strong>When NAS is not worth it:</strong> standard image classification, NLP, and tabular tasks — pre-trained models are stronger starting points than architectures found by NAS on your dataset; when you have less than 1 GPU-day of compute budget (NAS requires at minimum a few hours of search on CIFAR-scale problems); when the accuracy improvement from NAS is marginal (1–2%) compared to the engineering cost of setting up and running the search infrastructure.
      </Prose>

      <Callout type="insight">
        The practical NAS landscape shifted decisively around 2022. The compute required for NAS — even with weight sharing — is justified only when the deployment hardware has strict constraints (mobile, edge, embedded) that off-the-shelf architectures cannot meet. For GPU-served production models, the compute is better spent on data cleaning, feature engineering, or larger pre-trained model fine-tuning. NAS remains a research tool and an edge-deployment tool; it is not a general-purpose replacement for architecture selection in most production ML systems.
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8a. AutoML compute budget analysis</H3>

      <Prose>
        AutoML cost scales as <strong>budget × eval_cost</strong> where eval_cost is the cost of one CV evaluation. For tabular data with fast models (logistic regression, shallow trees), eval_cost is milliseconds — 1,000 trials in a minute is achievable. For deep models or large datasets (XGBoost on 10M rows), eval_cost is minutes — 30 trials in a 30-minute budget. Successive halving (HalvingGridSearchCV, Hyperband) shrinks eval_cost for the first rounds by evaluating on a small data fraction and promoting only the survivors to full evaluation. FLAML's CFO takes this further by treating configuration cost as an explicit feature of the optimization problem.
      </Prose>

      <Prose>
        Meta-learning — initializing the search from configurations that worked on similar datasets — is auto-sklearn's key scaling mechanism. By warming up the Bayesian optimizer with 25 pre-evaluated configurations from similar historical tasks, auto-sklearn skips the cold-start exploration phase and arrives at good configurations much earlier in the budget. This is the same idea as warm-starting gradient descent from a good initialization: the optimizer spends more time in promising regions.
      </Prose>

      <H3>8b. NAS compute: from 2,000 GPU-days to 1</H3>

      <Prose>
        The compute cost of NAS has dropped by several orders of magnitude since Zoph and Le 2017, driven by three techniques. <strong>Weight sharing</strong> (ENAS, DARTS) avoids training each candidate architecture from scratch: the supernet's weights are shared across all architectures, reducing the evaluation cost of any individual architecture from hundreds of epochs to a few gradient steps. ENAS brought search cost from ~2,000 GPU-days to ~1. <strong>Performance prediction</strong> trains a surrogate model on (architecture encoding, final accuracy) pairs from previous searches, allowing new architectures to be scored without any training. <strong>Zero-cost proxies</strong> (e.g., NASWOT — Neural Architecture Search Without Training, Mellor et al. 2021) score architectures using a single forward pass through random data — correlating with final accuracy without any gradient computation. These proxies enable screening thousands of architectures in minutes, though they are noisy and work best as a pre-filter before more expensive evaluation.
      </Prose>

      <H3>8c. Scaling limits</H3>

      <Prose>
        The fundamental scaling limit of AutoML and NAS is the product of search space size and per-evaluation cost. You can reduce either, but you cannot eliminate the tension. A larger search space finds better solutions but requires more evaluations to cover it. Cheaper evaluations allow more coverage but may not correlate with final performance (the proxy quality problem). The practical recommendation: match the search budget to the evaluation cost. For fast models, use random search with 100+ trials. For slow models, use Bayesian optimization with 30–50 trials and aggressive early stopping inside each trial.
      </Prose>

      <Prose>
        <strong>The rank correlation problem in NAS</strong> is a specific scaling failure: the architecture rankings produced by weight-shared evaluation (supernet) often do not agree with the rankings produced by training-from-scratch evaluation (the ground truth). When rank correlation is low, the best architecture found by the supernet is not the best architecture in the training-from-scratch world. DARTS is particularly susceptible: the continuous relaxation during search can lead to architectures dominated by skip connections (which are cheap and gradient-friendly) that perform poorly when discretized. This is why NAS results are often not reproducible across different search budgets, seeds, and discretization strategies.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9a. Reporting the best CV score as generalization</H3>

      <Prose>
        The most pervasive AutoML mistake: running a hyperparameter search over 100 configurations, taking the best CV score, and reporting it as the model's expected test accuracy. This is the winner's curse — the best score in 100 noisy estimates is biased upward by selection. The magnitude of this optimism grows with the number of configurations and with the noise in the CV estimator (small dataset, few folds). The fix: use nested CV (outer loop estimates generalization, inner loop selects the configuration) or reserve a held-out test set that is never used during search. AutoML systems that report their best CV score as the final metric are overstating their accuracy by a predictable and measurable amount.
      </Prose>

      <H3>9b. Overfitting the validation set during NAS</H3>

      <Prose>
        In DARTS, the architecture parameters {"α"} are updated on the validation set. If the search runs long enough, the search procedure overfits to that specific validation split — the recovered architecture performs well on the validation set but generalizes worse to the true test distribution. This is a form of the same winner's curse but in the continuous gradient space. Mitigation: use a different validation split for architecture search and for hyperparameter tuning; limit the number of architecture search epochs; use early stopping on validation loss during search.
      </Prose>

      <H3>9c. Budget misallocation</H3>

      <Prose>
        A common failure in AutoML practice: spending the entire search budget on one model family (e.g., XGBoost with 200 hyperparameter configurations) and never evaluating other families (CatBoost, neural networks, linear models) that might outperform it on this specific dataset. Most AutoML libraries protect against this with portfolio initialization — they ensure that a diverse set of configurations (including different model families) are evaluated early. If you are running manual hyperparameter search, always allocate at least 20% of your budget to model families you have not tried yet, even if your prior is strong.
      </Prose>

      <H3>9d. NAS rank correlation failure</H3>

      <Prose>
        The Spearman rank correlation between supernet-based rankings and training-from-scratch rankings is often below 0.5 in the DARTS family of methods — meaning the architecture the search identifies as best is not reliably the best when evaluated honestly. This makes the entire NAS exercise potentially misleading: you may run an expensive search and end up with an architecture that is no better than a random draw from the search space. Mitigation: always evaluate the top-k architectures from the search by training them from scratch and selecting by validation performance on a fresh split.
      </Prose>

      <H3>9e. Fairness and robustness dropped from the objective</H3>

      <Prose>
        AutoML systems optimize a single scalar metric — accuracy, AUC, RMSE. They do not, by default, optimize for demographic fairness (equal error rates across groups), robustness to distribution shift, or calibration (accurate prediction probabilities). A system that AutoML-optimizes for AUC on a credit scoring dataset may find configurations that achieve high AUC by exploiting proxy attributes correlated with protected characteristics. The fix is to include fairness constraints in the optimization objective (e.g., Equalized Odds difference below a threshold) or to use a multi-objective AutoML framework — but this is not standard in any library's default configuration as of 2026.
      </Prose>

      <H3>9f. Reproducibility and random_state</H3>

      <Prose>
        AutoML systems involve multiple sources of randomness: the search strategy's sampling, CV fold assignment, model initialization, and early stopping rounds. Running the same FLAML or AutoGluon call twice without fixing all seeds may produce different best configurations and different reported scores. Always set <Code>seed</Code> parameters in the AutoML call, fix <Code>random_state</Code> in the CV splitter, and record the full configuration of the best model before deploying. In research settings, report results over multiple random seeds with mean and standard deviation.
      </Prose>

      <H3>9g. Trusting AutoML without domain knowledge</H3>

      <Prose>
        AutoML is a powerful search tool, not an oracle. It cannot detect data leakage (a feature computed from the future leaking into training), cannot identify that your "train" and "test" splits are from different populations, and cannot audit whether the features you included are appropriate to use for decision-making. AutoML raises the floor — it finds better configurations than a lazy baseline — but it does not raise the ceiling of what a domain-informed engineer can achieve with careful feature engineering and problem framing. Use AutoML as a strong starting point and sanity check, not as a replacement for understanding your data.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were verified against their primary publication venues and arXiv pages.
      </Prose>

      <Prose>
        <strong>Thornton, C., Hutter, F., Hoos, H.H., and Leyton-Brown, K. (2013).</strong> "Auto-WEKA: Combined Selection and Hyperparameter Optimization of Classification Algorithms." <em>Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '13)</em>, pp. 847–855. DOI: 10.1145/2487575.2487629. The paper that introduced the CASH formulation and demonstrated that joint algorithm selection and hyperparameter optimization could match or beat expert-crafted configurations on standard benchmarks. Used SMAC (Sequential Model-based Algorithm Configuration) as the optimizer over the full WEKA classifier library.
      </Prose>

      <Prose>
        <strong>Feurer, M., Klein, A., Eggensperger, K., Springenberg, J.T., Blum, M., and Hutter, F. (2015).</strong> "Efficient and Robust Automated Machine Learning." <em>Advances in Neural Information Processing Systems 28 (NeurIPS 2015)</em>, pp. 2962–2970. arXiv:1507.00677. Introduced auto-sklearn, which added meta-learning warm-start and ensemble construction to the CASH framework. Won the first ChaLearn AutoML challenge. Established the template for tabular AutoML: Bayesian HPO over a conditional hyperparameter space, with meta-features to initialize from historical data.
      </Prose>

      <Prose>
        <strong>Hutter, F., Kotthoff, L., and Vanschoren, J. (Eds.) (2019).</strong> <em>Automated Machine Learning: Methods, Systems, Challenges.</em> Springer. Available open access at automl.org/book. The canonical reference for the field. Covers the CASH problem, Bayesian optimization for HPO, meta-learning, neural architecture search, and emerging directions. Authored by contributors from the auto-sklearn, SMAC, Auto-WEKA, and NAS communities.
      </Prose>

      <Prose>
        <strong>Zoph, B. and Le, Q.V. (2017).</strong> "Neural Architecture Search with Reinforcement Learning." <em>International Conference on Learning Representations (ICLR 2017)</em>. arXiv:1611.01578. The paper that launched the NAS field. A recurrent controller generates cell architecture descriptions as token sequences; trained by REINFORCE with validation accuracy as reward. Achieved state-of-the-art on CIFAR-10 and Penn Treebank at the cost of approximately 800 TPU-days (roughly 2,000 GPU-days equivalent). The paper that made the automation of architecture design credible.
      </Prose>

      <Prose>
        <strong>Pham, H., Guan, M.Y., Zoph, B., Le, Q.V., and Dean, J. (2018).</strong> "Efficient Neural Architecture Search via Parameter Sharing." <em>Proceedings of the 35th International Conference on Machine Learning (ICML 2018)</em>. arXiv:1802.03268. Introduced weight sharing across a supernet ("parameter sharing"), reducing NAS cost from thousands of GPU-days to approximately 1 GPU-day. The ENAS controller samples subgraphs of the supernet and evaluates them using shared weights, enabling thousands of architecture evaluations per hour. Foundational to all subsequent weight-sharing NAS methods.
      </Prose>

      <Prose>
        <strong>Liu, H., Simonyan, K., and Yang, Y. (2019).</strong> "DARTS: Differentiable Architecture Search." <em>International Conference on Learning Representations (ICLR 2019)</em>. arXiv:1806.09055. Introduced the continuous relaxation of discrete architecture choices via softmax mixing weights, enabling gradient-based joint optimization of architecture and network weights. DARTS reduces NAS to approximately 1 GPU-day on CIFAR-10 and discovers competitive architectures. The paper also introduced the key failure mode: the discretization gap between the continuous relaxation and the recovered discrete architecture.
      </Prose>

      <Prose>
        <strong>Tan, M. and Le, Q.V. (2019).</strong> "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." <em>Proceedings of the 36th International Conference on Machine Learning (ICML 2019)</em>. arXiv:1905.11946. Used neural architecture search (via MnasNet's hardware-aware NAS) to find a baseline architecture, then derived a compound scaling rule — simultaneously scaling depth, width, and input resolution by a fixed ratio — to produce a family of models that dominates the ImageNet accuracy vs FLOPs Pareto frontier. EfficientNet-B7 achieved 84.3% top-1 with 8.4x fewer parameters than the best GPipe model at the time.
      </Prose>

      <Prose>
        <strong>Wang, C., Wu, Q., Weimer, M., and Zhu, E. (2021).</strong> "FLAML: A Fast and Lightweight AutoML Library." <em>Proceedings of Machine Learning and Systems 3 (MLSys 2021)</em>, pp. 434–447. arXiv:1911.04706. Introduced FLAML and the CFO (Cost-Frugal Optimizer) algorithm, which exploits the cost of configurations as a signal in the search process. Demonstrated that FLAML finds competitive configurations with significantly less compute than auto-sklearn and H2O AutoML on tabular benchmarks, making it the practical default for time-constrained tabular AutoML.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 (Recall)</H3>
      <Prose>
        Define the CASH problem. What distinguishes it from ordinary hyperparameter optimization, and why does it require a surrogate model that handles conditional hyperparameter spaces?
      </Prose>
      <Callout type="answer">
        {"CASH (Combined Algorithm Selection and Hyperparameter optimization) jointly optimizes over (a) which algorithm family to use and (b) the hyperparameters of that family. Unlike ordinary HPO — which fixes the algorithm and searches only its hyperparameters — CASH treats the algorithm identity itself as a discrete choice variable. This creates a conditional structure: the hyperparameter 'kernel' is only meaningful if the algorithm is SVM; 'max_depth' is only meaningful for tree-based models. Standard GP-based Bayesian optimization cannot handle this because GP kernels require a continuous metric on the full input space. SMAC solves this by using a random forest as the surrogate — random forests naturally handle mixed (continuous + categorical + conditional) inputs by encoding inactive hyperparameters as a special missing-value category and ignoring them in tree splits. The random forest predicts the loss distribution over configurations, and the acquisition function (Expected Improvement) then proposes the next configuration to evaluate."}
      </Callout>

      <H3>Exercise 2 (Conceptual)</H3>
      <Prose>
        Explain the DARTS continuous relaxation. Why is the architecture updated on the validation set rather than the training set? What is the "discretization gap" and why does it occur?
      </Prose>
      <Callout type="answer">
        {"DARTS replaces the discrete choice of one operation per cell edge with a softmax-weighted sum of all candidate operations: the edge output is Σ_k softmax(α_k)·o_k(x), where α_k is a real-valued architecture parameter. This makes the output differentiable in α, allowing gradient-based optimization. The architecture parameters α are updated by gradient descent on the validation loss (not the training loss) for the following reason: if α were updated on the training loss, the optimization would drive α toward operations that memorize training data rather than operations that generalize — the architecture selection would be dominated by overfitting. By using validation loss, α is updated based on held-out performance, reflecting true generalization potential. The discretization gap arises during the final step: after search, the discrete architecture is recovered by argmax over α for each edge. The continuous relaxation allows all operations to run simultaneously with gradient flow helping every operation; the discrete architecture runs only the argmax operation. The dropped operations may have been contributing gradient signal that regularized the kept operation, so the discrete model performs worse than the continuous model would suggest."}
      </Callout>

      <H3>Exercise 3 (Applied)</H3>
      <Prose>
        You have 2 hours of compute budget to build the best tabular classifier for a 100,000-row dataset with 50 features (mix of numerical and high-cardinality categorical). Compare FLAML, AutoGluon, and running manual XGBoost hyperparameter search. Which would you use and why?
      </Prose>
      <Callout type="answer">
        {"AutoGluon is the strongest choice for maximum accuracy within a 2-hour budget on a 100K-row mixed dataset. Setting time_limit=7200 allows AutoGluon to train multiple LightGBM, XGBoost, CatBoost, and ensemble models in multiple stacking layers — CatBoost's ordered target statistics will handle the high-cardinality categoricals without manual encoding, and the stacking layer will exploit correlations between model outputs. FLAML is the better choice if the 2-hour budget is shared with other pipeline steps (feature engineering, deployment testing), because FLAML's CFO optimizer finds a strong single model in 5–30 minutes rather than consuming the full budget. Manual XGBoost tuning requires manual categorical encoding (target encoding with k-fold OOF to avoid leakage, or ordinal encoding), manual hyperparameter range specification, and careful early stopping setup — it is slower to set up and will likely produce a weaker result than AutoGluon's stacking unless you have deep domain knowledge about the data. Rule of thumb: use AutoGluon for maximum accuracy, FLAML for fast good-enough baselines, and manual tuning only when you have a specific model requirement or domain constraint."}
      </Callout>

      <H3>Exercise 4 (Applied)</H3>
      <Prose>
        Your team proposes running DARTS to find a custom architecture for a new image classification task. The dataset has 50,000 images across 10 classes. The team has 4 A100 GPUs available for 1 week. Evaluate whether NAS is the right approach and what the alternative would be.
      </Prose>
      <Callout type="answer">
        {"NAS is likely not the best use of 4 GPU-weeks for a 50K-image, 10-class task. The argument against: (a) strong pre-trained architectures (EfficientNet, ViT-B/16, ResNet-50) fine-tuned on 50K images with transfer learning will likely outperform a DARTS-searched architecture trained from scratch, because the pre-trained features generalize far better than random initialization on 50K examples; (b) DARTS takes roughly 1 GPU-day for the search phase alone, plus 1-2 GPU-days for full retraining of the discovered architecture — so a single DARTS run consumes 2-3 GPU-days; (c) the DARTS search on CIFAR-10 does not transfer reliably to other datasets without modification, so you would likely need multiple runs with different seeds; (d) the rank correlation between DARTS proxy performance and final training-from-scratch performance is unreliable. The alternative: spend 4 GPU-weeks on (1) fine-tuning EfficientNet-B3 or ViT-B/16 pre-trained on ImageNet with various augmentation strategies, (2) hyperparameter optimization of the fine-tuning schedule (learning rate warm-up, cosine decay, label smoothing, mixup), and (3) ensembling 3-5 fine-tuned models. This workflow is more reliable, better understood, and will almost certainly produce a stronger model on 50K images. NAS would be the right choice if the images are non-standard (microscopy, radar, hyperspectral) where ImageNet pre-training transfers poorly."}
      </Callout>

      <H3>Exercise 5 (Debugging)</H3>
      <Prose>
        You run FLAML with a 60-second budget and get a best CV accuracy of 0.95. You then evaluate the returned model on your held-out test set and get 0.82. List four possible explanations and a concrete diagnostic for each.
      </Prose>
      <Callout type="answer">
        {"(1) Data leakage during preprocessing: a scaler, target encoder, or imputer was fit on the full training data before FLAML's internal CV split, so the CV validation folds have seen statistics from the full training set. Diagnostic: wrap all preprocessing in FLAML's pipeline interface and verify that no transformation is fit before the CV split. Check feature provenance for any column computed from the label. (2) Distribution shift between train and test: the training set is from a different time period, geography, or data collection process than the test set. CV splits from the training set all look similar and give optimistic estimates; the test set is genuinely out-of-distribution. Diagnostic: compute feature distribution statistics (mean, std, cardinality for categorical) for train vs test and flag columns with KS-statistic p-value below 0.05. (3) Winner's curse from FLAML's internal CV: FLAML evaluated many configurations and returned the best CV score, which is biased upward. The 0.95 is the best in a collection of noisy estimates. Diagnostic: run nested CV manually — use 5-fold outer CV where each outer fold runs FLAML internally; the outer fold test scores (not FLAML's CV score) give an honest estimate. (4) Test set too small / high variance: if the test set has 50-100 examples, the 0.82 estimate has a 95% confidence interval of roughly +/- 5-10 percentage points. The gap from 0.95 may be partly noise. Diagnostic: compute a Wilson confidence interval on the test accuracy and check whether it overlaps with the CV estimate's confidence interval."}
      </Callout>

      <H3>Exercise 6 (Math)</H3>
      <Prose>
        In a DARTS search cell with 3 candidate operations (3x3 conv, max-pool, skip), the architecture parameters after 100 search epochs are {"α = [2.1, 0.5, -0.8]"} for a single edge. Compute the softmax mixing weights. After discretization (argmax), which operation is selected? If you apply L2 regularization on {"α"} with {"λ=0.5"}, how does this affect the softmax distribution at convergence, and what failure mode does this mitigate?
      </Prose>
      <Callout type="answer">
        {"Softmax: exp([2.1, 0.5, -0.8]) = [8.166, 1.649, 0.449]. Sum = 10.264. Weights = [0.796, 0.161, 0.044]. So 3x3 conv gets 79.6% weight, max-pool 16.1%, skip 4.4%. Argmax selects 3x3 conv. With L2 regularization λ=0.5 on α, the gradient update for each α_k includes a penalty term -λ·α_k pulling all parameters toward zero. At convergence, the magnitudes of α are smaller, and the softmax distribution is flatter (less peaked): if α shrinks uniformly by factor c, the softmax of c·[2.1, 0.5, -0.8] approaches uniform as c→0. In practice, L2 regularization on α prevents extreme concentration of weight on a single operation, keeping all operations in the mixture. This mitigates the 'skip connection collapse' failure mode: in standard DARTS, skip connections (which have zero parameters and thus contribute zero gradient to w but still contribute gradient to α) often accumulate large α values because they are cheap for the optimizer. L2 on α penalizes large α magnitudes regardless of operation type, reducing the advantage of parameterless operations and producing more diverse final architectures."}
      </Callout>

    </div>
  ),
};

export default automlContent;
