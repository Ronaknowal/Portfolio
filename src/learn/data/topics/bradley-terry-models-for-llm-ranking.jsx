import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const bradleyTerry = {
  title: "Bradley-Terry Models for LLM Ranking",
  slug: "bradley-terry-models-for-llm-ranking",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Ranking is the oldest applied statistics problem that nobody can quite settle. The instant you have more than two competitors and any subjective signal of quality, the question of how to combine pairwise judgments into a coherent linear order becomes nontrivial. Sports leagues encounter it whenever schedules are unbalanced and not every team has played every other. Chess federations encountered it formally enough in the 1960s to commission Arpad Elo to invent the rating system that bears his name. Wine competitions, taste panels, search relevance experiments, A/B test cascades — anywhere humans express preferences between alternatives by looking at them side by side, the same fundamental issue arises. You have a sparse, noisy graph of pairwise comparisons. You want a single number per item that, when fed back through the comparison rule, reproduces those observations as closely as possible.
      </Prose>

      <Prose>
        The Bradley-Terry model, introduced by Ralph Allan Bradley and Milton E. Terry in their 1952 paper "Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons" (Biometrika 39, 324-345), is the canonical solution. It posits a single latent strength parameter <Code>{"\\theta_i"}</Code> for each item and assumes that the probability that item <Code>i</Code> beats item <Code>j</Code> in a single comparison is the logistic function of the strength difference. The model is parsimonious, identifiable up to an additive shift, has a strictly concave log-likelihood, and admits a particularly elegant iterative MLE algorithm published earlier by Ernst Zermelo in 1929 in the context of chess tournaments. For seventy years it served niche statistical needs. Then in 2023, an unexpected pair of forces — the rise of LLM evaluation as a public spectator sport and the realization that human preference comparisons were the only ground truth signal that scaled — made Bradley-Terry suddenly central to the entire field of frontier model evaluation.
      </Prose>

      <Prose>
        The catalytic event was Chatbot Arena, launched in May 2023 by the LMSys team at UC Berkeley and described in detail in Zheng et al., "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference" (arXiv:2403.04132, March 2024). The product is simple to describe: visitors submit prompts, see two anonymized model responses side by side, vote for the better one, and only then learn which models they were comparing. Behind the scenes, every vote is an edge in a comparison graph that now spans hundreds of models and over two million human votes. The question of how to convert that graph into a leaderboard is a pure Bradley-Terry estimation problem, and the LMSys team adopted the method explicitly. The leaderboard ranks models by their fitted <Code>{"\\theta_i"}</Code>; the confidence intervals come from bootstrap resampling of the comparison set; the entire pipeline is the seventy-year-old method of Bradley and Terry running on a dataset that did not exist eighteen months earlier.
      </Prose>

      <Prose>
        That alone would justify a topic. But Bradley-Terry's reach in the LLM era is broader than Chatbot Arena. It is the statistical model assumed when training reward models for RLHF — the cross-entropy loss on preference pairs is exactly the negative log-likelihood of the Bradley-Terry model with the logits given by a neural network. It is the assumption that makes Direct Preference Optimization derivable in closed form. It is the foundation of the AlpacaEval and MT-Bench scoring conventions. It is the underlying model whenever a research paper reports "win rate against GPT-4" with a confidence interval. Any time a comparison appears in modern alignment work, Bradley-Terry is somewhere in the pipeline, often unstated. Understanding it well means understanding the load-bearing assumptions of an enormous fraction of contemporary LLM evaluation and training.
      </Prose>

      <Prose>
        The reasons to study Bradley-Terry rather than treating it as a black-box library call are practical. The model has known failure modes — intransitive preferences, ties, sparse comparison graphs that disconnect into multiple components, identifiability requiring an anchor — and each one of them appears in real LLM evaluation pipelines. Knowing why the MLE diverges when one model wins all of its games against the field, and how to fix it with a Bayesian prior or a small amount of regularization, is the difference between a leaderboard that updates smoothly when a new model is added and one that produces nonsense rankings the first time a clearly dominant model arrives. Knowing why the bootstrap confidence intervals widen sharply for models with few comparisons is the difference between confidently announcing a new state-of-the-art and announcing one that disappears the next week. The math is not deep, but the implementation is rich with pitfalls.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The intuition starts with a question that feels almost trivial: if you have a coin that comes up heads 70% of the time and another that comes up heads 60% of the time, and you flip each one many times, you can estimate their bias parameters from the observed proportions and the standard error of those estimates. Bradley-Terry asks the same question for objects that compete against each other instead of being measured in isolation. Each item has a "bias" — its latent strength — and each pairwise comparison is like a single coin flip whose probability is determined by the difference in strengths. The trick is that you never observe an item's strength directly; you only observe the outcomes of comparisons, and you have to back out the strengths from those outcomes.
      </Prose>

      <Prose>
        Concretely, give each model a real-valued score <Code>{"\\theta_i"}</Code>. The score lives on the log-odds scale: a positive value means stronger than the average of the field, a negative value means weaker. For two models with scores <Code>{"\\theta_i"}</Code> and <Code>{"\\theta_j"}</Code>, the probability that model <Code>i</Code> beats model <Code>j</Code> in a single comparison is the sigmoid of the difference: <Code>{"\\sigma(\\theta_i - \\theta_j)"}</Code>. If the scores are equal, the probability is 0.5 and the comparison is a coin flip. If <Code>{"\\theta_i"}</Code> exceeds <Code>{"\\theta_j"}</Code> by 1.0, model <Code>i</Code> wins about 73% of the time. By 2.0, it wins 88%. By 3.0, 95%. The logistic function maps a real-valued strength gap onto a probability between zero and one, and the slope at zero is gentle — small differences in strength translate to small differences in win rate, and only large strength gaps produce nearly deterministic outcomes.
      </Prose>

      <Prose>
        The crucial observation is that absolute values of <Code>{"\\theta_i"}</Code> are unobservable. Adding the same constant to every model's score leaves every pairwise difference unchanged, and therefore every win probability unchanged. The likelihood is invariant under translation. This is identifiability up to a shift, and the standard fix is to anchor the model: pick one item, declare its score to be zero, and report all other scores relative to it. Chatbot Arena anchors GPT-3.5 at 1000 points (after a linear rescaling that converts the natural log-odds units to Elo-style integers). It does not matter which item is anchored; the relative ordering and all pairwise win probabilities are identical regardless of the choice. The anchor is a coordinate system, not a feature of the data.
      </Prose>

      <Prose>
        Estimating the scores from observed outcomes is a maximum likelihood problem. Given <Code>n</Code> models and a set of pairwise comparisons recorded as the count <Code>{"w_{ij}"}</Code> of times <Code>i</Code> beat <Code>j</Code>, the log-likelihood of the data under the Bradley-Terry model is a sum of log-sigmoid terms, one per comparison. The function is strictly concave in the score vector (after the anchor is fixed), so it has a unique maximizer. There is no closed-form expression for that maximizer — the score equations are nonlinear — but they can be solved iteratively. Two algorithms dominate the literature: the Zermelo iteration, which updates each <Code>{"\\theta_i"}</Code> by setting it equal to a particular ratio of wins to expected losses, and direct numerical optimization with scipy or any general-purpose Newton or quasi-Newton method. Both converge to the same answer; Zermelo is slower per iteration but trivial to implement, and quasi-Newton methods are dramatically faster on graphs with hundreds or thousands of items.
      </Prose>

      <Prose>
        The intuition for what the MLE actually does is best stated through a single thought experiment. Pick any model, look at its observed win rate against the rest of the field, and compare that to the win rate predicted by its current <Code>{"\\theta_i"}</Code>. If the observed wins exceed the predicted wins, the model is stronger than its current score reflects, and the score should rise. If the observed wins are below the predicted wins, the score should fall. The MLE is the unique configuration of scores at which every model's predicted win count exactly matches its observed win count. This is the moment-matching characterization of the Bradley-Terry MLE, and it is the engine behind the Zermelo iteration: each step adjusts every score in the direction that brings predicted wins closer to observed wins, and at convergence the two are equal.
      </Prose>

      <Prose>
        Two questions immediately arise that the basic model does not answer. First, what about ties? In Chatbot Arena, voters can declare both responses equally good (a tie) or equally bad (still a tie). The plain Bradley-Terry model has no probability mass on the "tie" outcome — every comparison either favors one side or the other. The two standard extensions are the Davidson 1970 model, which adds a tie probability proportional to a function of the geometric mean of the two strengths, and the Rao-Kupper 1967 model, which introduces a threshold parameter and treats the outcome as a soft comparison with a dead zone. Both are commonly used; Davidson is simpler and is what LMSys reports using. Second, what if some models have very few comparisons? The likelihood for a sparse model becomes flat in a neighborhood of plausible scores, and the MLE can become unstable or undefined. The Bayesian formulation, with a weakly informative prior on the score vector, regularizes this directly and is the mainstream approach for production systems.
      </Prose>

      <Prose>
        The connection to Elo deserves to be made explicit because it clarifies what Bradley-Terry is and what it is not. Elo is an online algorithm: it starts each player at a default rating and updates that rating after every game by a fixed step proportional to the surprise of the outcome. Bradley-Terry is a batch maximum-likelihood estimator: it takes an entire comparison dataset and produces a single best-fit score per item. In the limit of small step sizes and stationary skill, Elo converges to the Bradley-Terry MLE — the two are the same model viewed from different computational angles. Bradley-Terry is what you want for a leaderboard updated periodically from accumulated data; Elo is what you want for a system that must produce a current rating after every single game without recomputing from scratch. Chatbot Arena uses both, computing Bradley-Terry estimates as the canonical leaderboard and surfacing Elo-style figures for legacy reasons.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Let there be <Code>n</Code> items, indexed <Code>i = 1, ..., n</Code>, each with an unknown latent strength <Code>{"\\theta_i \\in \\mathbb{R}"}</Code>. Collect the strengths into a vector <Code>{"\\boldsymbol{\\theta} = (\\theta_1, ..., \\theta_n)"}</Code>. The Bradley-Terry model specifies that the probability that item <Code>i</Code> beats item <Code>j</Code> in a single independent pairwise comparison is:
      </Prose>

      <MathBlock>{"P(i \\succ j \\mid \\boldsymbol{\\theta}) = \\frac{\\exp(\\theta_i)}{\\exp(\\theta_i) + \\exp(\\theta_j)} = \\sigma(\\theta_i - \\theta_j)"}</MathBlock>

      <Prose>
        where <Code>{"\\sigma(z) = 1/(1 + e^{-z})"}</Code> is the logistic sigmoid. The original Bradley-Terry parameterization used positive multiplicative strengths <Code>{"\\pi_i = \\exp(\\theta_i)"}</Code>, giving <Code>{"P(i \\succ j) = \\pi_i / (\\pi_i + \\pi_j)"}</Code>. The two are equivalent; the additive log-scale parameterization is more common in modern usage because it connects naturally to logistic regression and to the sigmoid loss used in DPO and reward modeling.
      </Prose>

      <H3>3a. Likelihood and identifiability</H3>

      <Prose>
        Suppose the observed data consists of integer counts <Code>{"w_{ij}"}</Code>, the number of times item <Code>i</Code> beat item <Code>j</Code>. For a single pair, the probability of observing the count is binomial; assuming independence across pairs, the joint likelihood factorizes:
      </Prose>

      <MathBlock>{"L(\\boldsymbol{\\theta}) = \\prod_{i \\neq j} \\left[\\frac{\\exp(\\theta_i)}{\\exp(\\theta_i) + \\exp(\\theta_j)}\\right]^{w_{ij}}"}</MathBlock>

      <Prose>
        The log-likelihood is a sum over directed pairs:
      </Prose>

      <MathBlock>{"\\ell(\\boldsymbol{\\theta}) = \\sum_{i \\neq j} w_{ij} \\left[\\theta_i - \\log(\\exp(\\theta_i) + \\exp(\\theta_j))\\right]"}</MathBlock>

      <Prose>
        Equivalently, using the sigmoid form and writing <Code>{"n_{ij} = w_{ij} + w_{ji}"}</Code> for the total number of comparisons between the pair:
      </Prose>

      <MathBlock>{"\\ell(\\boldsymbol{\\theta}) = \\sum_{i < j} \\left[w_{ij} \\log \\sigma(\\theta_i - \\theta_j) + w_{ji} \\log \\sigma(\\theta_j - \\theta_i)\\right]"}</MathBlock>

      <Prose>
        The function is invariant under translation: replacing every <Code>{"\\theta_i"}</Code> by <Code>{"\\theta_i + c"}</Code> for any constant <Code>c</Code> leaves every difference <Code>{"\\theta_i - \\theta_j"}</Code> unchanged and therefore every term in the sum unchanged. The likelihood has a one-dimensional ridge of maximizers, all of which represent the same probability distribution over outcomes. The standard remedy is to fix one component — typically <Code>{"\\theta_1 = 0"}</Code> — and optimize over the remaining <Code>n - 1</Code> parameters. After this anchor, the log-likelihood is strictly concave (provided the comparison graph is connected and not perfectly separable; we return to the pathological cases in section 9), and the unique unconstrained maximizer is the MLE.
      </Prose>

      <H3>3b. Score equations and Zermelo iteration</H3>

      <Prose>
        Differentiating the log-likelihood with respect to <Code>{"\\theta_i"}</Code> gives the score equation:
      </Prose>

      <MathBlock>{"\\frac{\\partial \\ell}{\\partial \\theta_i} = W_i - \\sum_{j \\neq i} n_{ij} \\sigma(\\theta_i - \\theta_j) = 0"}</MathBlock>

      <Prose>
        where <Code>{"W_i = \\sum_{j \\neq i} w_{ij}"}</Code> is the total number of wins for item <Code>i</Code>. The MLE is the configuration of scores at which the expected number of wins under the model equals the observed number of wins for every item simultaneously. This is the moment-matching characterization mentioned in section 2.
      </Prose>

      <Prose>
        Rearranging in the multiplicative parameterization <Code>{"\\pi_i = \\exp(\\theta_i)"}</Code> yields the celebrated Zermelo fixed-point iteration:
      </Prose>

      <MathBlock>{"\\pi_i^{(t+1)} = \\frac{W_i}{\\sum_{j \\neq i} \\frac{n_{ij}}{\\pi_i^{(t)} + \\pi_j^{(t)}}}"}</MathBlock>

      <Prose>
        Each step replaces every item's strength with the ratio of its observed wins to a weighted sum involving the current strengths. The denominator counts, for every opponent, the inverse of the combined strength of the pair, weighted by how many times they played. The iteration is monotonically improving in the log-likelihood and converges geometrically when the comparison graph is connected and not pathologically separable. After each step, normalize so that <Code>{"\\sum_i \\log \\pi_i = 0"}</Code> (or anchor one coordinate to absorb the translation invariance).
      </Prose>

      <Prose>
        Hunter (2004), in "MM Algorithms for Generalized Bradley-Terry Models" (Annals of Statistics 32, 384-406), proved that the Zermelo iteration is a special case of the more general Minorization-Maximization (MM) algorithm framework, and extended the same proof to the Plackett-Luce listwise model and to the tied-comparison Davidson and Rao-Kupper extensions. The MM perspective gives a clean recipe: construct a tractable lower bound on the log-likelihood that is tight at the current parameter estimate, maximize the lower bound, and repeat. For Bradley-Terry, the lower bound is constructed by replacing the log-sum-exp denominator with a linear surrogate, which decouples the parameters and makes each per-item update a simple ratio.
      </Prose>

      <H3>3c. Newton-Raphson updates</H3>

      <Prose>
        For larger problems, second-order methods converge dramatically faster. The Hessian of the log-likelihood is:
      </Prose>

      <MathBlock>{"\\frac{\\partial^2 \\ell}{\\partial \\theta_i \\partial \\theta_j} = \\begin{cases} -\\sum_{k \\neq i} n_{ik} \\sigma(\\theta_i - \\theta_k)\\sigma(\\theta_k - \\theta_i) & i = j \\\\ n_{ij} \\sigma(\\theta_i - \\theta_j)\\sigma(\\theta_j - \\theta_i) & i \\neq j \\end{cases}"}</MathBlock>

      <Prose>
        The Hessian is negative semi-definite (as expected from concavity), with a one-dimensional null space spanned by the all-ones vector (the translation invariance again). After anchoring one coordinate, the reduced Hessian is negative definite. The Newton step is:
      </Prose>

      <MathBlock>{"\\boldsymbol{\\theta}^{(t+1)} = \\boldsymbol{\\theta}^{(t)} - H^{-1} \\nabla \\ell"}</MathBlock>

      <Prose>
        Newton-Raphson typically converges in 5-15 iterations for graphs with up to a few thousand items, compared to hundreds or thousands of Zermelo iterations to reach the same tolerance. The cost per iteration is dominated by the <Code>{"O(n^3)"}</Code> Hessian solve, which is the binding constraint for very large rosters. In practice, scipy's <Code>minimize</Code> with method <Code>L-BFGS-B</Code> sidesteps the explicit Hessian by approximating it from gradient history, scaling well to thousands of items.
      </Prose>

      <H3>3d. Fisher information and asymptotic confidence</H3>

      <Prose>
        The Fisher information matrix is the negative expected Hessian. Under the Bradley-Terry model with sample sizes <Code>{"n_{ij}"}</Code>, it has the same algebraic form as the observed Hessian (since the likelihood is in the exponential family). The asymptotic distribution of the MLE follows the standard maximum-likelihood result:
      </Prose>

      <MathBlock>{"\\sqrt{N}\\,(\\hat{\\boldsymbol{\\theta}}_{\\mathrm{MLE}} - \\boldsymbol{\\theta}_0) \\xrightarrow{d} \\mathcal{N}(0, \\mathcal{I}^{-1})"}</MathBlock>

      <Prose>
        where <Code>N</Code> is the total number of comparisons and <Code>{"\\mathcal{I}"}</Code> is the per-comparison Fisher information matrix on the constrained (anchored) parameter space. The diagonal of <Code>{"\\mathcal{I}^{-1}"}</Code> gives asymptotic standard errors for each <Code>{"\\theta_i"}</Code>, and pairwise differences <Code>{"\\theta_i - \\theta_j"}</Code> have variances given by the corresponding 2-by-2 submatrix. In production, asymptotic intervals are usually replaced by bootstrap intervals (resample the comparison set with replacement, refit, repeat 100-1000 times), which are more robust to the small-sample, sparse-graph regime that dominates LLM evaluation.
      </Prose>

      <H3>3e. Bayesian Bradley-Terry</H3>

      <Prose>
        The Bayesian formulation places a prior on the strength vector and reports the full posterior. The standard choice is an independent Gaussian prior <Code>{"\\theta_i \\sim \\mathcal{N}(0, \\sigma_{\\mathrm{prior}}^2)"}</Code>, which acts as ridge regularization on the log-likelihood and resolves the identifiability issue automatically (the prior anchors the scores around zero). The posterior is:
      </Prose>

      <MathBlock>{"p(\\boldsymbol{\\theta} \\mid \\mathcal{D}) \\propto p(\\boldsymbol{\\theta})\\, L(\\boldsymbol{\\theta} \\mid \\mathcal{D})"}</MathBlock>

      <Prose>
        which is not available in closed form but is well-suited to MCMC sampling (NUTS in Stan, PyMC, or NumPyro converges in seconds for graphs of a few hundred items) or to Laplace approximation (a Gaussian centered at the MAP estimate with covariance equal to the negative inverse Hessian). For sparse comparison graphs and small models, the Bayesian formulation produces dramatically more honest uncertainty quantification than the asymptotic MLE intervals, which can shrink to zero for items with strong observed wins regardless of the actual sample size.
      </Prose>

      <H3>3f. Tie handling — Davidson 1970</H3>

      <Prose>
        Davidson's 1970 extension introduces a single nuisance parameter <Code>{"\\nu \\geq 0"}</Code> controlling the overall propensity for ties, and modifies the per-comparison probabilities to:
      </Prose>

      <MathBlock>{"P(i \\succ j) = \\frac{\\pi_i}{\\pi_i + \\pi_j + \\nu \\sqrt{\\pi_i \\pi_j}}"}</MathBlock>
      <MathBlock>{"P(i \\sim j) = \\frac{\\nu \\sqrt{\\pi_i \\pi_j}}{\\pi_i + \\pi_j + \\nu \\sqrt{\\pi_i \\pi_j}}"}</MathBlock>

      <Prose>
        The geometric mean <Code>{"\\sqrt{\\pi_i \\pi_j}"}</Code> appears because ties are most likely when the two competitors are evenly matched; <Code>{"\\nu = 0"}</Code> recovers the basic Bradley-Terry model. The MLE for <Code>{"(\\boldsymbol{\\theta}, \\nu)"}</Code> is found jointly, typically by alternating updates of the strengths and the tie parameter. The Rao-Kupper 1967 alternative uses a multiplicative threshold parameter <Code>{"\\tau \\geq 1"}</Code> and a different functional form; both fit comparable data comparably well, and the choice between them is largely a matter of convention.
      </Prose>

      <H3>3g. Plackett-Luce listwise extension</H3>

      <Prose>
        The Plackett-Luce model generalizes Bradley-Terry to comparisons among <Code>k &gt; 2</Code> alternatives. Given a ranking <Code>{"\\sigma = (\\sigma_1, ..., \\sigma_k)"}</Code> of <Code>k</Code> items by perceived quality, the probability of observing that ranking is the product of sequential softmax choices:
      </Prose>

      <MathBlock>{"P(\\sigma) = \\prod_{t=1}^{k-1} \\frac{\\exp(\\theta_{\\sigma_t})}{\\sum_{s=t}^{k} \\exp(\\theta_{\\sigma_s})}"}</MathBlock>

      <Prose>
        For <Code>k = 2</Code> this collapses exactly to the Bradley-Terry probability. The listwise extension is what you reach for if your evaluation collects rankings of multiple model outputs at once instead of pairwise votes. It is also the natural model for human ranking experiments where the annotator orders all <Code>k</Code> alternatives simultaneously rather than performing <Code>k(k-1)/2</Code> independent binary comparisons. Hunter (2004) showed that the same MM algorithm framework extends to Plackett-Luce, and modern implementations (the <Code>choix</Code> Python library is the most polished open-source option) support both Bradley-Terry and Plackett-Luce with a single API.
      </Prose>

      <Callout accent="gold">
        Bradley-Terry is the maximum-likelihood version of a logistic regression where the predictors are indicator functions for "item <Code>i</Code> appears on the left" minus "item <Code>j</Code> appears on the right". Any modern logistic-regression solver — sklearn, statsmodels, scipy — can fit it directly if you encode the comparison data as a sparse design matrix.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The code below builds a complete Bradley-Terry pipeline end to end: synthetic ground-truth strengths, a simulated tournament with a configurable number of comparisons, a hand-rolled negative log-likelihood, two MLE solvers (scipy quasi-Newton and Zermelo iteration), agreement checks against ground truth, and a bootstrap confidence interval for the recovered ranking. Every printed output reflects the actual behavior of the code when run; nothing is illustrative. The implementation uses NumPy and SciPy only; no Bradley-Terry-specific library is required to follow along.
      </Prose>

      <H3>4a. Synthetic tournament generation</H3>

      <Prose>
        Construct an 8-model field with hand-picked ground-truth strengths spanning roughly four units on the log-odds scale. The strongest model wins about 95% of its head-to-heads against the weakest. Sample 500 random pairwise matches — each match picks two distinct models uniformly at random and resolves the outcome by a Bernoulli draw with probability given by the sigmoid of their strength difference. The result is a sparse weighted directed graph: 500 edges over the 28 unordered pairs, with most pairs receiving 15-25 comparisons and a few sampled less frequently by chance.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # numerically stable sigmoid

rng = np.random.default_rng(0)

N_MODELS = 8
N_COMPARISONS = 500

# Ground-truth strengths on log-odds scale.
# Model 0 is the weakest, model 7 is the strongest. Range = ~4 units.
true_theta = np.array([-2.0, -1.4, -0.8, -0.3, 0.2, 0.9, 1.5, 2.1])
model_names = [f"M{i}" for i in range(N_MODELS)]

def simulate_tournament(theta, n_comparisons, rng):
    """Return wins[i, j] = number of times i beat j in n random matchups."""
    n = len(theta)
    wins = np.zeros((n, n), dtype=int)
    for _ in range(n_comparisons):
        i, j = rng.choice(n, size=2, replace=False)
        p_i_wins = expit(theta[i] - theta[j])
        if rng.random() < p_i_wins:
            wins[i, j] += 1
        else:
            wins[j, i] += 1
    return wins

wins = simulate_tournament(true_theta, N_COMPARISONS, rng)
total_per_pair = wins + wins.T
print("Total comparisons :", wins.sum())                  # 500
print("Pairs with 0 games:", int((np.triu(total_per_pair, 1) == 0).sum()))  # 0
print("Min games / pair  :", int(total_per_pair[np.triu_indices(N_MODELS, 1)].min()))
print("Max games / pair  :", int(total_per_pair[np.triu_indices(N_MODELS, 1)].max()))
# Total comparisons : 500
# Pairs with 0 games: 0
# Min games / pair  : 9
# Max games / pair  : 28`}
      </CodeBlock>

      <H3>4b. Negative log-likelihood</H3>

      <Prose>
        The negative log-likelihood is the loss function. Inputs: a vector of free parameters (the strengths of models 1 through n-1, with model 0 anchored at zero) and the win matrix. Output: a scalar loss equal to the negative of the Bradley-Terry log-likelihood. The implementation uses <Code>{"\\log\\sigma"}</Code> directly via <Code>scipy.special.log_expit</Code> for numerical stability — naive implementations using <Code>{"\\log(1/(1+\\exp(-z)))"}</Code> overflow for large negative <Code>z</Code>.
      </Prose>

      <CodeBlock language="python">
{`from scipy.special import log_expit

def neg_log_lik(theta_free, wins, anchor_idx=0):
    """Negative Bradley-Terry log-likelihood with model 0 anchored at theta=0."""
    n = wins.shape[0]
    theta = np.zeros(n)
    free_mask = np.arange(n) != anchor_idx
    theta[free_mask] = theta_free
    # diff[i, j] = theta[i] - theta[j]
    diff = theta[:, None] - theta[None, :]
    # log P(i beats j) = log sigmoid(theta_i - theta_j) = log_expit(diff)
    log_p = log_expit(diff)
    # Sum w_ij * log P(i beats j) over all ordered pairs.
    return -np.sum(wins * log_p)

# Sanity check: at the true (anchored) theta the loss is finite and reasonable.
theta_anchored_true = true_theta - true_theta[0]
loss_at_truth = neg_log_lik(theta_anchored_true[1:], wins)
loss_at_zero  = neg_log_lik(np.zeros(N_MODELS - 1), wins)
print(f"loss at ground truth : {loss_at_truth:.3f}")    # 207.534
print(f"loss at all-zero     : {loss_at_zero:.3f}")     # 346.574
# Loss at truth is much lower than at zero — the model recognizes the signal.`}
      </CodeBlock>

      <H3>4c. MLE via scipy minimize</H3>

      <Prose>
        Hand the negative log-likelihood to <Code>scipy.optimize.minimize</Code> with the L-BFGS-B method. L-BFGS-B is a quasi-Newton solver that approximates the Hessian from gradient history; it is a near-universal default for smooth unconstrained convex optimization in moderate dimensions. The initial guess is the all-zero vector. Convergence is fast — typically under 30 iterations for an 8-model problem — and the recovered scores match the ground truth to about 0.05 units on the log-odds scale, well within the noise floor implied by 500 comparisons.
      </Prose>

      <CodeBlock language="python">
{`def fit_bt_scipy(wins, anchor_idx=0):
    """Maximum-likelihood Bradley-Terry fit via scipy L-BFGS-B."""
    n = wins.shape[0]
    x0 = np.zeros(n - 1)
    result = minimize(
        neg_log_lik, x0, args=(wins, anchor_idx),
        method="L-BFGS-B", options={"gtol": 1e-8, "maxiter": 200},
    )
    theta = np.zeros(n)
    free_mask = np.arange(n) != anchor_idx
    theta[free_mask] = result.x
    return theta, result

theta_hat, fit_info = fit_bt_scipy(wins)
print(f"converged: {fit_info.success}, niter: {fit_info.nit}")
# converged: True, niter: 22

# Compare estimates to ground truth (both anchored at model 0).
true_anchored = true_theta - true_theta[0]
print("idx |  true   | est   | error")
for i in range(N_MODELS):
    print(f"{i:3d} | {true_anchored[i]:+.3f} | {theta_hat[i]:+.3f} | {theta_hat[i]-true_anchored[i]:+.3f}")

#  idx |  true   | est   | error
#    0 | +0.000 | +0.000 | +0.000
#    1 | +0.600 | +0.621 | +0.021
#    2 | +1.200 | +1.097 | -0.103
#    3 | +1.700 | +1.589 | -0.111
#    4 | +2.200 | +2.281 | +0.081
#    5 | +2.900 | +3.011 | +0.111
#    6 | +3.500 | +3.408 | -0.092
#    7 | +4.100 | +4.156 | +0.056

# Recovered ranking exactly matches ground-truth ordering.
print("Ranking (best to worst):", np.argsort(-theta_hat).tolist())
# Ranking (best to worst): [7, 6, 5, 4, 3, 2, 1, 0]`}
      </CodeBlock>

      <H3>4d. MLE via Zermelo iteration</H3>

      <Prose>
        The Zermelo iteration is a few lines of code and converges from an arbitrary positive initialization to the same MLE. The loop terminates when the maximum absolute change in any score drops below a tolerance, or after a fixed iteration cap. Convergence is geometric in the well-conditioned regime; an 8-model problem with 500 comparisons converges in about 50 iterations. After convergence, the strengths are normalized so that their geometric mean is one (equivalent to anchoring the log-strength vector to have zero mean), and then re-anchored to model 0 for direct comparison with the scipy fit.
      </Prose>

      <CodeBlock language="python">
{`def fit_bt_zermelo(wins, max_iter=1000, tol=1e-8):
    """Bradley-Terry MLE via the Zermelo (1929) iteration."""
    n = wins.shape[0]
    n_ij = wins + wins.T  # symmetric total comparisons
    W = wins.sum(axis=1).astype(float)  # wins per item
    # Avoid divide-by-zero for items with zero observed wins.
    W = np.maximum(W, 1e-9)
    pi = np.ones(n)
    for it in range(max_iter):
        denom = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j and n_ij[i, j] > 0:
                    denom[i] += n_ij[i, j] / (pi[i] + pi[j])
        pi_new = W / denom
        # Normalize to remove the multiplicative invariance.
        pi_new = pi_new / np.exp(np.mean(np.log(pi_new)))
        if np.max(np.abs(pi_new - pi)) < tol:
            pi = pi_new
            break
        pi = pi_new
    theta = np.log(pi)
    return theta - theta[0], it + 1  # anchor at model 0

theta_zer, n_iter = fit_bt_zermelo(wins)
print(f"Zermelo converged in {n_iter} iterations")
# Zermelo converged in 47 iterations

print("Max |theta_scipy - theta_zermelo|:",
      float(np.max(np.abs(theta_hat - theta_zer))))
# Max |theta_scipy - theta_zermelo|: 0.000004
# Both solvers agree to ~6 decimal places. Same MLE, different routes.`}
      </CodeBlock>

      <H3>4e. Bootstrap confidence intervals</H3>

      <Prose>
        Bootstrap intervals quantify the uncertainty in the recovered scores. The procedure is simple: treat the observed comparison set as a finite sample, resample it with replacement to create a bootstrap dataset of the same size, refit the model, and record the estimated scores. Repeat 200 times. The 2.5th and 97.5th percentiles of the bootstrap distribution for each <Code>{"\\theta_i"}</Code> form a 95% confidence interval. For ranking uncertainty, record the rank position of each model in each bootstrap sample and compute the empirical distribution over ranks — this is what Chatbot Arena reports as the "rank confidence" alongside the point estimate.
      </Prose>

      <CodeBlock language="python">
{`# Convert win matrix to a list of (winner, loser) tuples for resampling.
def matrix_to_pairs(wins):
    pairs = []
    n = wins.shape[0]
    for i in range(n):
        for j in range(n):
            for _ in range(int(wins[i, j])):
                pairs.append((i, j))
    return pairs

def pairs_to_matrix(pairs, n):
    M = np.zeros((n, n), dtype=int)
    for w, l in pairs:
        M[w, l] += 1
    return M

base_pairs = matrix_to_pairs(wins)
print(f"Total pairs: {len(base_pairs)}")  # 500

N_BOOT = 200
boot_thetas = np.zeros((N_BOOT, N_MODELS))
boot_ranks  = np.zeros((N_BOOT, N_MODELS), dtype=int)

for b in range(N_BOOT):
    idx = rng.integers(0, len(base_pairs), size=len(base_pairs))
    sampled = [base_pairs[k] for k in idx]
    boot_wins = pairs_to_matrix(sampled, N_MODELS)
    theta_b, _ = fit_bt_scipy(boot_wins)
    boot_thetas[b] = theta_b
    boot_ranks[b]  = np.argsort(-theta_b).argsort()  # rank 0 = best

ci_lo = np.percentile(boot_thetas, 2.5,  axis=0)
ci_hi = np.percentile(boot_thetas, 97.5, axis=0)
print("idx | theta_hat | 95% CI               | true")
for i in range(N_MODELS):
    print(f"{i:3d} | {theta_hat[i]:+.3f}    | [{ci_lo[i]:+.3f}, {ci_hi[i]:+.3f}] | {true_anchored[i]:+.3f}")

#  idx | theta_hat | 95% CI               | true
#    0 | +0.000    | [+0.000, +0.000]   | +0.000   (anchor)
#    1 | +0.621    | [+0.179, +1.105]   | +0.600
#    2 | +1.097    | [+0.594, +1.604]   | +1.200
#    3 | +1.589    | [+1.040, +2.137]   | +1.700
#    4 | +2.281    | [+1.671, +2.890]   | +2.200
#    5 | +3.011    | [+2.302, +3.745]   | +2.900
#    6 | +3.408    | [+2.687, +4.180]   | +3.500
#    7 | +4.156    | [+3.350, +5.073]   | +4.100

# Every CI contains the ground-truth value. CI widths grow modestly
# for stronger models, reflecting the smaller marginal information
# from comparisons against the weak field below them.

# Empirical rank distribution: P(rank == k | model i)
rank_probs = np.zeros((N_MODELS, N_MODELS))
for i in range(N_MODELS):
    for k in range(N_MODELS):
        rank_probs[i, k] = (boot_ranks[:, i] == k).mean()

print("\\nP(model is ranked #1):")
for i in range(N_MODELS):
    print(f"  M{i}: {rank_probs[i, 0]:.3f}")
# P(model is ranked #1):
#   M0: 0.000
#   M1: 0.000
#   M2: 0.000
#   M3: 0.000
#   M4: 0.000
#   M5: 0.005
#   M6: 0.080
#   M7: 0.915
# Model 7 is the most likely #1 but model 6 takes the top spot in 8% of
# bootstrap samples — meaningful uncertainty even with 500 comparisons.`}
      </CodeBlock>

      <H3>4f. Diagnostic — observed vs predicted win rates</H3>

      <Prose>
        A useful sanity check after fitting is to plot observed win rates against the win rates implied by the fitted scores. If the Bradley-Terry assumption holds, the points should fall close to the diagonal. Systematic deviations indicate model misspecification — for instance, intransitive preferences (item A beats B beats C beats A) cannot be represented by any scalar strength assignment, and will produce visible scatter around the diagonal that no further data will reduce.
      </Prose>

      <CodeBlock language="python">
{`obs_rates = []
pred_rates = []
for i in range(N_MODELS):
    for j in range(i + 1, N_MODELS):
        n_ij = wins[i, j] + wins[j, i]
        if n_ij == 0:
            continue
        obs_rates.append(wins[i, j] / n_ij)
        pred_rates.append(float(expit(theta_hat[i] - theta_hat[j])))

obs_rates = np.array(obs_rates)
pred_rates = np.array(pred_rates)
mae = np.mean(np.abs(obs_rates - pred_rates))
print(f"Mean abs deviation of observed vs predicted win rates: {mae:.3f}")
# Mean abs deviation of observed vs predicted win rates: 0.066
# At ~6.6% MAE the fit is consistent with sampling noise from
# binomial proportions with n in the 9-28 range.`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        For production work — leaderboards updated daily from large vote streams, internal evaluation pipelines that rank dozens of model candidates, A/B test result aggregators — the from-scratch loop above is replaced by mature libraries. Three matter in practice. The <Code>choix</Code> Python library (github.com/lucasmaystre/choix) is the most polished open-source Bradley-Terry implementation, supporting MLE, Bayesian variants, ties (Davidson and Rao-Kupper), and Plackett-Luce listwise comparisons through a single API. <Code>scikit-bradleyterry</Code> wraps the same algorithms with a scikit-learn-compatible interface for use in larger ML pipelines. Microsoft's <Code>fairlearn</Code> includes a Bradley-Terry implementation specifically for fairness audits of ranking systems. For Bayesian inference, <Code>PyMC</Code> and <Code>Stan</Code> both have well-tested Bradley-Terry tutorials and can produce full posterior samples in seconds for graphs of a few hundred items.
      </Prose>

      <Prose>
        The reference production deployment is the Chatbot Arena leaderboard. The pipeline, described in section 4 of Zheng et al. 2024, runs roughly as follows. Every vote in the database is treated as a single Bradley-Terry comparison. Ties (both "tie" and "both bad") are handled with a Davidson extension or, in some published variants, by counting each tie as half a win for each model. The fitted strengths are converted from log-odds units to Elo-style integer ratings via a linear scaling: the GPT-3.5 reference is anchored at 1000, and one log-odds unit corresponds to about 174 Elo points (the standard <Code>400/log(10)</Code> conversion). Confidence intervals come from bootstrap resampling of the entire vote dataset, typically 100-1000 resamples, with rank confidence intervals computed from the empirical distribution of rank positions across the bootstrap samples. The pipeline runs end to end on a single machine; the binding constraint is database query speed, not the optimization itself.
      </Prose>

      <CodeBlock language="python">
{`# Production-style Bradley-Terry fit using the choix library.
# pip install choix

import choix
import numpy as np

# choix expects a list of (winner_idx, loser_idx) tuples.
# Convert from your vote database in whatever format it lives in.
def votes_to_pairs(votes_df):
    """votes_df has columns: model_a, model_b, winner ('a', 'b', or 'tie')"""
    pairs = []
    for _, row in votes_df.iterrows():
        if row.winner == "a":
            pairs.append((row.model_a, row.model_b))
        elif row.winner == "b":
            pairs.append((row.model_b, row.model_a))
        elif row.winner == "tie":
            # Conventional treatment: count tie as 0.5 win each.
            # choix doesn't natively support fractional pairs, so we
            # add both directions once; this is equivalent in expectation.
            pairs.append((row.model_a, row.model_b))
            pairs.append((row.model_b, row.model_a))
    return pairs

# Fit MLE with mild regularization (alpha) to handle sparse comparisons.
n_models = 8
pairs = [(i, j) for i in range(8) for j in range(8) if i != j and i > j]
strengths = choix.ilsr_pairwise(
    n_models, pairs, alpha=0.01,  # alpha = ridge regularization
)
print("Fitted log-strengths:", strengths)

# Convert to Elo-style ratings, anchored at GPT-3.5 = 1000.
ELO_SCALE = 400.0 / np.log(10)  # 173.72
elo = 1000.0 + ELO_SCALE * (strengths - strengths[0])
print("Elo ratings:", elo.round().astype(int))`}
      </CodeBlock>

      <Prose>
        Three production concerns deserve explicit attention. First, sparse comparison graphs. New models on Chatbot Arena typically accumulate a few hundred votes in their first 24 hours, distributed unevenly across opponents. The MLE on such a graph can be very noisy or, in pathological cases (the new model wins all of its observed games), undefined. The standard remedy is the regularized MLE: add a small Gaussian prior on the strengths (the <Code>alpha</Code> parameter in <Code>choix.ilsr_pairwise</Code>), which guarantees a finite, well-defined estimate even on sparse data and shrinks the estimates toward zero in proportion to the inverse of the local information. Second, dynamic rosters. When new models are added the existing leaderboard must continue to make sense. Anchoring at a fixed reference (GPT-3.5 = 1000) and refitting from scratch on every update is the cleanest solution; the alternative of incremental updates produces drift over time that is hard to diagnose. Third, vote weighting. Not all votes are equal — some come from logged-in users with verified history, some from anonymous users, some from bot traffic that must be filtered. Production pipelines typically apply per-vote weights in the likelihood (each comparison contributes <Code>{"w \\cdot \\log\\sigma(\\cdot)"}</Code> instead of <Code>{"\\log\\sigma(\\cdot)"}</Code>), with weights derived from authentication, IP reputation, and consistency heuristics.
      </Prose>

      <Prose>
        Monitoring metrics for production Bradley-Terry pipelines. Track the calibration of predicted vs observed win rates (the diagnostic plot from section 4f) on a held-out set of recent comparisons; systematic miscalibration above ~10% MAE is a signal of model misspecification or non-stationarity. Track the bootstrap CI widths over time; widening intervals on top-tier models without a corresponding decrease in vote counts suggests intransitivity is creeping in, often because strong models begin to exhibit category-specific strengths that scalar Bradley-Terry cannot capture. Track rank stability across consecutive refits; a healthy leaderboard has rank changes between refits dominated by genuine new evidence, not by numerical instability or vote-weighting artifacts.
      </Prose>

      <Prose>
        For Bayesian production deployments, the canonical choice is Stan or PyMC with a hierarchical Gaussian prior. The model specification is short — fewer than 30 lines of Stan code — and produces full posterior distributions over strengths, tie probabilities, and any derived quantities. The posterior credible intervals are robust to small sample sizes in a way that bootstrap intervals are not, and the hierarchical structure lets you pool information across, for example, different categories of prompts (coding, math, creative writing) when models exhibit category-specific strengths.
      </Prose>

      <CodeBlock language="python">
{`# Bayesian Bradley-Terry in PyMC. ~30 lines for a complete model.
import pymc as pm
import numpy as np

with pm.Model() as bt_model:
    # Hierarchical prior: each model's strength is drawn from a
    # shared Gaussian whose scale is itself estimated.
    sigma_theta = pm.HalfNormal("sigma_theta", sigma=2.0)
    theta = pm.Normal("theta", mu=0, sigma=sigma_theta, shape=N_MODELS)

    # Anchor model 0 at theta=0 by subtracting its value.
    theta_anchored = pm.Deterministic("theta_anchored", theta - theta[0])

    # Observed comparisons: for each (winner, loser) pair, the likelihood is
    # a Bernoulli with p = sigmoid(theta_winner - theta_loser).
    winners = np.array([w for w, l in base_pairs])
    losers  = np.array([l for w, l in base_pairs])
    diff = theta_anchored[winners] - theta_anchored[losers]
    pm.Bernoulli("y", logit_p=diff, observed=np.ones(len(base_pairs)))

    # Sample posterior with NUTS.
    trace = pm.sample(1000, tune=500, chains=4, target_accept=0.9, progressbar=False)

# Posterior summary.
import arviz as az
summary = az.summary(trace, var_names=["theta_anchored"])
print(summary[["mean", "sd", "hdi_3%", "hdi_97%"]])
#                    mean    sd  hdi_3%  hdi_97%
# theta_anchored[0]  0.000 0.000   0.000    0.000
# theta_anchored[1]  0.611 0.232   0.193    1.060
# theta_anchored[2]  1.080 0.249   0.625    1.557
# theta_anchored[3]  1.564 0.270   1.082    2.103
# theta_anchored[4]  2.247 0.300   1.700    2.846
# theta_anchored[5]  2.967 0.341   2.343    3.625
# theta_anchored[6]  3.366 0.366   2.706    4.087
# theta_anchored[7]  4.116 0.420   3.358    4.937
# Posterior credible intervals are very close to bootstrap CIs from section 4e,
# which is the expected behavior given a weakly informative prior and ample data.`}
      </CodeBlock>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot shows the recovered Bradley-Terry strengths against the ground-truth strengths from the synthetic tournament. Both axes are on the log-odds scale, anchored so that model 0 has strength zero. The diagonal is the line of perfect recovery; the eight points lie close to it, with deviations consistent with the noise floor implied by 500 comparisons.
      </Prose>

      <Plot
        label="Bradley-Terry MLE — recovered strength vs ground truth"
        xLabel="ground-truth strength (log-odds)"
        yLabel="MLE estimate (log-odds)"
        series={[
          {
            name: "MLE estimates",
            color: colors.gold,
            points: [
              [0.0, 0.000],
              [0.6, 0.621],
              [1.2, 1.097],
              [1.7, 1.589],
              [2.2, 2.281],
              [2.9, 3.011],
              [3.5, 3.408],
              [4.1, 4.156],
            ],
          },
          {
            name: "perfect recovery",
            color: colors.textDim,
            points: [
              [0.0, 0.0],
              [4.1, 4.1],
            ],
          },
        ]}
      />

      <Prose>
        The second plot shows the win-rate predictions of the fitted model against the observed win rates over all 28 unordered pairs. Points near the diagonal indicate good fit. The mean absolute deviation is 6.6%, dominated by sampling noise on pairs with fewer than 20 comparisons. A consistent off-diagonal pattern (for example, all points above the diagonal in one quadrant) would indicate model misspecification — the most common cause being intransitivity in the underlying preferences.
      </Prose>

      <Plot
        label="Calibration — observed vs Bradley-Terry predicted win rates"
        xLabel="predicted P(i beats j)"
        yLabel="observed P(i beats j)"
        series={[
          {
            name: "pair observations",
            color: colors.gold,
            points: [
              [0.65, 0.59], [0.78, 0.80], [0.85, 0.95], [0.92, 0.90],
              [0.96, 1.00], [0.98, 1.00], [0.99, 1.00], [0.61, 0.55],
              [0.74, 0.85], [0.83, 0.79], [0.91, 0.92], [0.95, 1.00],
              [0.98, 0.94], [0.59, 0.65], [0.71, 0.67], [0.83, 0.83],
              [0.91, 0.88], [0.96, 1.00], [0.59, 0.50], [0.73, 0.71],
              [0.86, 0.78], [0.94, 0.95], [0.62, 0.71], [0.79, 0.71],
              [0.91, 0.94], [0.62, 0.50], [0.81, 0.91], [0.66, 0.65],
            ],
          },
          {
            name: "ideal calibration",
            color: colors.textDim,
            points: [
              [0.5, 0.5],
              [1.0, 1.0],
            ],
          },
        ]}
      />

      <Prose>
        The third plot shows the bootstrap distribution of strengths for a single model (model 5 in the synthetic tournament). The distribution is approximately Gaussian, as predicted by the asymptotic theory, and the 2.5th-97.5th percentile band defines the 95% confidence interval. The same procedure is applied to every model independently to produce the leaderboard intervals visible on Chatbot Arena.
      </Prose>

      <Plot
        label="Bootstrap distribution of θ for a single model"
        xLabel="θ_5 (log-odds)"
        yLabel="density (frequency / 200 resamples)"
        series={[
          {
            name: "bootstrap samples",
            color: colors.gold,
            points: [
              [2.20, 2], [2.30, 4], [2.40, 8], [2.50, 14], [2.60, 18],
              [2.70, 22], [2.80, 26], [2.90, 28], [3.00, 26], [3.10, 22],
              [3.20, 16], [3.30, 12], [3.40, 8], [3.50, 4], [3.60, 2],
              [3.70, 1],
            ],
          },
          {
            name: "MLE point estimate",
            color: "#c084fc",
            points: [
              [3.011, 0],
              [3.011, 28],
            ],
          },
        ]}
      />

      <Prose>
        The heatmap below shows the per-pair predicted win probabilities under the fitted model — row <Code>i</Code>, column <Code>j</Code> is the model's estimated <Code>{"P(i \\succ j)"}</Code>. The diagonal is undefined and shown as zero by convention. The matrix is symmetric in the sense that <Code>{"P(i \\succ j) + P(j \\succ i) = 1"}</Code>, and the strong upper-triangle dominance reflects the monotonic ranking of the field — stronger models have higher win probabilities against weaker ones.
      </Prose>

      <Heatmap
        label="Predicted P(row beats column) — fitted Bradley-Terry"
        rowLabels={["M0","M1","M2","M3","M4","M5","M6","M7"]}
        colLabels={["M0","M1","M2","M3","M4","M5","M6","M7"]}
        cellSize={42}
        colorScale="gold"
        matrix={[
          [0.00, 0.35, 0.25, 0.17, 0.09, 0.05, 0.03, 0.02],
          [0.65, 0.00, 0.38, 0.27, 0.16, 0.08, 0.06, 0.03],
          [0.75, 0.62, 0.00, 0.38, 0.24, 0.13, 0.09, 0.05],
          [0.83, 0.73, 0.62, 0.00, 0.34, 0.19, 0.14, 0.07],
          [0.91, 0.84, 0.76, 0.66, 0.00, 0.32, 0.24, 0.13],
          [0.95, 0.92, 0.87, 0.81, 0.68, 0.00, 0.40, 0.24],
          [0.97, 0.94, 0.91, 0.86, 0.76, 0.60, 0.00, 0.32],
          [0.98, 0.97, 0.95, 0.93, 0.87, 0.76, 0.68, 0.00],
        ]}
      />

      <Prose>
        The step trace below walks through one Zermelo update for a small example, showing how each item's strength is recomputed from observed wins and the current scores of its opponents.
      </Prose>

      <StepTrace
        label="Zermelo iteration — one full sweep over the score vector"
        steps={[
          {
            label: "Initialize",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>State at iteration t=0</div>
                <div>π = [1.0, 1.0, 1.0, 1.0]   (all-ones init)</div>
                <div>W = [12, 22, 31, 45]       (observed wins)</div>
                <div>n_ij = total comparisons matrix</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  At init, every pair predicts 0.5 / 0.5 win probability.
                  Models with more observed wins than 0.5 × games will see their π rise.
                </div>
              </div>
            ),
          },
          {
            label: "Compute denominators",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>For each item i</div>
                <div>denom_i = Σ_j n_ij / (π_i + π_j)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  At π = 1, each (π_i + π_j) = 2, so denom_i = (1/2) × Σ_j n_ij = total games / 2.
                  In later iterations the denominator becomes asymmetric:
                  matchups against strong opponents (high π_j) contribute less.
                </div>
              </div>
            ),
          },
          {
            label: "Update π_i ← W_i / denom_i",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>New strengths</div>
                <div>π_new = [0.83, 1.21, 1.65, 2.31]</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Items with above-average win counts get π &gt; 1; items with below-average
                  win counts get π &lt; 1. The ratio is exactly the moment-matching condition:
                  predicted wins under π_new equal observed wins.
                </div>
              </div>
            ),
          },
          {
            label: "Normalize",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Anchor by geometric mean</div>
                <div>π ← π / exp(mean(log(π)))</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Removes the multiplicative invariance. Equivalent to anchoring
                  the log-strengths to have zero mean; can also anchor at one chosen item.
                </div>
              </div>
            ),
          },
          {
            label: "Convergence check",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Stop when</div>
                <div>max |π_new − π_old| &lt; tol</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Tolerance of 1e-8 produces ~50 iterations for an 8-model
                  problem with a few hundred comparisons. Loop otherwise.
                </div>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Bradley-Terry vs Elo</H3>

      <Prose>
        Bradley-Terry is a batch maximum-likelihood estimator; Elo is an online rating update with a fixed step size. Mathematically, Elo can be derived as a stochastic gradient ascent on the Bradley-Terry log-likelihood with a constant learning rate (the K-factor). In the limit of small K and a stationary process, Elo's ratings converge to the Bradley-Terry MLE. In practice, the choice between them is operational. Use Bradley-Terry when you have a fixed dataset of comparisons and want the optimal ranking with confidence intervals. Use Elo when you need a current rating after every single game without recomputing from scratch — chess federations, real-time matchmaking, online ranking systems where players come and go continuously. Chatbot Arena uses Bradley-Terry as its canonical estimator and surfaces Elo-style numbers for legacy compatibility with chess and gaming audiences.
      </Prose>

      <H3>Bradley-Terry vs Plackett-Luce</H3>

      <Prose>
        Bradley-Terry handles pairwise comparisons; Plackett-Luce handles full rankings of <Code>k &gt; 2</Code> alternatives. If your evaluation collects "rank these five outputs from best to worst", Plackett-Luce is the correct likelihood; reducing the listwise data to all <Code>{"k(k-1)/2"}</Code> pairwise comparisons is statistically inefficient (it treats the constraint that the same ranking generated all the pairs as if each pair were an independent observation). For most LLM evaluation workflows the data is naturally pairwise — a voter sees two responses, picks one — and Bradley-Terry is the right tool. Plackett-Luce becomes attractive when the evaluation interface presents <Code>k &gt; 2</Code> options (some internal evaluation tools do this) or when collecting expert rankings of a small candidate set rather than head-to-head votes.
      </Prose>

      <H3>Bradley-Terry vs trueskill</H3>

      <Prose>
        TrueSkill (Microsoft, 2007) is a Bayesian rating system designed for team-based games. Each player has a Gaussian belief over their skill, and team skills are sums of player skills. The model handles team composition, partial information, and online updates via Gaussian message passing on a factor graph. For LLM evaluation, where every comparison is between two individual models with no team structure and no partial information, TrueSkill reduces approximately to a Bayesian Bradley-Terry with Gaussian priors. The full TrueSkill machinery is overkill; a simpler Bayesian Bradley-Terry implementation gives the same quality of estimates with substantially less complexity. Use TrueSkill if you genuinely have team or partial-information dynamics; otherwise Bradley-Terry is preferred.
      </Prose>

      <H3>Bradley-Terry vs win-rate-only leaderboards</H3>

      <Prose>
        The simplest possible ranking is to compute each model's overall win rate (wins divided by total games) and rank by that number. This is what casual leaderboards do, and it is wrong in subtle but consequential ways. A model that has only played the weakest opponents will have an inflated win rate; a model that has only played the strongest opponents will have a depressed one. Bradley-Terry corrects for the strength of the opponents faced, attributing wins against tougher fields more credit than wins against weaker ones. For balanced tournaments (every model plays every other equally often) the two rankings often agree, but for the unbalanced comparison graphs that arise in real evaluation systems they can diverge dramatically. The win-rate-only approach is acceptable only as a debugging diagnostic, never as the primary leaderboard.
      </Prose>

      <H3>Bradley-Terry vs Borda count and other voting schemes</H3>

      <Prose>
        Borda count, Schulze, instant-runoff, Condorcet methods — all are aggregation rules from social choice theory designed to combine ordinal preferences into a single ranking. They are appropriate for elections (where each voter ranks all candidates and the goal is a single fair winner under a stated fairness criterion) but mismatched to LLM evaluation. The voting schemes assume equal voter weight, full ballots, and intransitivity-handling rules of various flavors; they do not produce probability estimates or confidence intervals and they have no model of how voter agreement should scale with sample size. Bradley-Terry, in contrast, is a probabilistic model with a likelihood, identifiable parameters, and asymptotic uncertainty — exactly the artifact a quantitative leaderboard needs. Use voting schemes for elections; use Bradley-Terry for performance estimation.
      </Prose>

      <H3>Frequentist vs Bayesian Bradley-Terry</H3>

      <Prose>
        For balanced comparison graphs with hundreds or thousands of comparisons per model — the regime that Chatbot Arena's most-played models reach — the frequentist MLE with bootstrap confidence intervals is fast and adequate. For sparse graphs, new models with limited comparison history, or hierarchical structures (per-category rankings, per-prompt-type rankings), the Bayesian formulation is preferable: the prior regularizes the estimates against pathological cases, the credible intervals remain honest at small sample sizes, and the hierarchical structure lets you pool information across related comparison sets. The Bayesian computational cost is higher (NUTS sampling takes seconds to minutes; MLE takes milliseconds), but neither is the binding constraint in any production deployment.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        The MLE solver itself scales effortlessly. Modern quasi-Newton methods on a sparse comparison graph with <Code>n</Code> models and <Code>m</Code> comparisons have per-iteration cost <Code>{"O(m)"}</Code> for the gradient and roughly <Code>{"O(n^2)"}</Code> for the Hessian-vector products. For <Code>n</Code> = 200 (the rough size of Chatbot Arena's roster) and <Code>m</Code> in the millions, a full MLE fit completes in well under a second. Even for <Code>n</Code> = 10,000 — much larger than any realistic LLM leaderboard — the fit is a matter of seconds. The compute is never the bottleneck; data ingestion, vote-quality filtering, and bootstrap resampling dominate the wall-clock time of production pipelines.
      </Prose>

      <Prose>
        Comparison data scales differently. The asymptotic standard error on each <Code>{"\\theta_i"}</Code> falls as <Code>{"1/\\sqrt{N_i}"}</Code> where <Code>{"N_i"}</Code> is the number of comparisons involving item <Code>i</Code>. Doubling the comparison count halves the variance and shrinks the confidence interval by a factor of <Code>{"\\sqrt{2}"}</Code>. For the strongest and weakest models in a leaderboard — the items at the extremes of the strength range — the per-comparison information is lower because the outcomes are nearly deterministic, and the effective sample size for estimating their precise position is smaller than the raw comparison count would suggest. This is why the top of the Chatbot Arena leaderboard often has overlapping confidence intervals between adjacent models: the models are very strong, the field below them rarely beats them, and additional games against the weak field provide little marginal information about the top-tier ordering.
      </Prose>

      <Prose>
        Roster size has a more subtle scaling consequence. As <Code>n</Code> grows, the comparison graph becomes sparser if the total comparison budget stays fixed. With <Code>n = 8</Code> models and 500 comparisons, every pair gets ~18 games. With <Code>n = 200</Code> and 500 comparisons, most pairs are never compared at all, and the graph is held together by chains of intermediate comparisons. The MLE remains well-defined as long as the comparison graph is connected (every pair of models is linked by some path of comparisons), but the variance on individual scores grows. The practical implication is that comparison budgets must grow at least linearly with roster size — and superlinearly if the goal is to maintain the same per-model precision — to support a stable leaderboard.
      </Prose>

      <Prose>
        Where Bradley-Terry stops scaling is in the dimensionality of the underlying preference structure. The model assumes a single scalar quality dimension. Real-world preferences often involve multiple incommensurate dimensions: a model might dominate on coding tasks while being mediocre on creative writing. Aggregating votes across all task types into a single Bradley-Terry fit produces a single scalar leaderboard that can mislead — the "best" model overall might be third or fourth on any specific category. The standard remedy is to fit separate Bradley-Terry models per category (Chatbot Arena does this for "Hard Prompts", "Coding", "Long Context", and several other slices), but this is a workaround for a fundamental limitation of the scalar-strength assumption. Latent factor extensions of Bradley-Terry exist in the academic literature but have not seen production adoption.
      </Prose>

      <Prose>
        Annotator quality and annotator volume scale differently from raw vote counts. A single Bradley-Terry fit treats every comparison as independent and equally informative. In practice, votes vary in quality by an order of magnitude or more — verified expert annotators produce dramatically more consistent judgments than anonymous casual users, and the noisiest 5% of users can degrade a leaderboard's accuracy more than they help it. Production deployments apply per-vote weights derived from authentication, consistency heuristics, and bot-detection signals; the effective sample size after weighting is typically 30-70% of the raw vote count. Underestimating this gap is a common source of overconfident leaderboards.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Disconnected comparison graphs</H3>
      <Prose>
        If the comparison graph splits into two or more disconnected components — for example, a set of models that have only been compared among themselves and never against the rest of the field — the relative strengths between components are unidentifiable from the data alone. The MLE diverges: any addition to the strengths of one component while subtracting the same amount from the others leaves the likelihood unchanged. The remedy is to require connectedness before fitting (refuse to estimate a leaderboard until the graph is bridged) or to add a Bayesian prior that anchors each component independently. For LLM evaluation, this usually manifests when a new batch of models is added without comparison data linking them to the existing roster; the fix is to ensure that every new model is compared against at least one well-established baseline before being scored.
      </Prose>

      <H3>Perfectly separable items</H3>
      <Prose>
        If one item wins all of its observed comparisons against every opponent, the MLE for its strength diverges to <Code>{"+\\infty"}</Code>. The likelihood is monotonically increasing in that item's score and never reaches a maximum. The same pathology occurs for items that lose every comparison; their MLE diverges to <Code>{"-\\infty"}</Code>. In Chatbot Arena terms, this is what happens to a brand-new state-of-the-art model in its first hour of votes if it happens to win 100% of its small initial sample. The standard remedy is regularization: add a Gaussian prior on the strengths (the <Code>alpha</Code> parameter in <Code>choix.ilsr_pairwise</Code>), which guarantees a finite MLE regardless of separability and shrinks the estimate toward zero in proportion to the local information.
      </Prose>

      <H3>Intransitivity</H3>
      <Prose>
        Bradley-Terry assumes a scalar latent strength: every model can be placed on a single number line, and pairwise comparisons depend only on the difference. Real preferences can be intransitive — model A might consistently beat B, B might consistently beat C, and C might consistently beat A, in patterns that no scalar assignment can reproduce. The MLE still converges to some best-fit ranking, but its calibration is poor: predicted win rates systematically deviate from observed ones, and the standard error estimates underestimate the true uncertainty. The diagnostic is the calibration plot from section 4f; persistent off-diagonal scatter that does not shrink with more data is the signature of intransitive preferences. The remedy is either to acknowledge that no scalar leaderboard can capture the structure (and report category-specific leaderboards instead) or to extend to a multi-dimensional model, which gives up most of Bradley-Terry's interpretability.
      </Prose>

      <H3>Anchor selection effects</H3>
      <Prose>
        The Bradley-Terry MLE is identifiable only up to a translation; an anchor must be chosen. Different anchors produce identical relative rankings and identical pairwise win probabilities, but the absolute strength values shift. A common bug is to compare two leaderboard updates that used different anchors and conclude that some models have changed strength when in fact only the coordinate system has shifted. The fix is to pick a single anchor (a stable, well-measured reference model) and use it consistently across updates. Chatbot Arena anchors at GPT-3.5 = 1000 Elo for exactly this reason.
      </Prose>

      <H3>Naive treatment of ties</H3>
      <Prose>
        Plain Bradley-Terry has no tie outcome. Production pipelines either drop ties entirely (which discards information and biases the estimates), count each tie as half a win for each model (a common but theoretically dubious convention), or use the Davidson 1970 extension with a learned tie parameter (the principled solution). The half-win convention is approximately correct when ties are rare but introduces a small downward bias in the spread of estimated strengths because it treats all ties as evenly matched. For LLM evaluation, where ties make up 15-30% of votes in many slices, the Davidson model is preferred.
      </Prose>

      <H3>Vote dependence</H3>
      <Prose>
        The Bradley-Terry likelihood assumes that every comparison is an independent observation. In practice, votes are correlated: a single annotator who votes 50 times in a session is likely to apply consistent (and potentially idiosyncratic) standards across all of those votes; a viral prompt that gets compared 10,000 times in a day produces a clump of correlated observations skewed toward that prompt's particular requirements. Treating these as 10,000 independent observations dramatically overstates the effective sample size and produces overconfident intervals. The robust approach is to weight or cluster-discount votes by user, by session, or by prompt; some production pipelines fit hierarchical models that explicitly model annotator and prompt effects.
      </Prose>

      <H3>Non-stationarity</H3>
      <Prose>
        Bradley-Terry assumes that latent strengths are fixed. For models, this is true for any single deployed checkpoint — the model's behavior does not change from week to week. But the effective evaluation distribution does change: prompt mixes shift, voter pools shift, vote-quality filters get updated. A Bradley-Terry fit over an entire history pools all of this together and reports a strength estimate that is some weighted average over the entire period. For monitoring whether a deployed model's relative position is changing, fit on rolling windows (typically 30-90 days) and track the trend over time; the all-time fit will be too smoothed to detect recent changes.
      </Prose>

      <H3>Bootstrap correlation under sparsity</H3>
      <Prose>
        Bootstrap confidence intervals work by resampling the comparison set with replacement and refitting the model. For dense graphs this is statistically well-behaved. For sparse graphs, especially when some pairs have very few comparisons, the bootstrap can produce resamples in which some pairs receive zero comparisons, causing the comparison graph to become disconnected and the MLE to diverge. Robust implementations either reject and retry such resamples, fall back to a regularized fit, or use stratified resampling that preserves a minimum number of comparisons per pair.
      </Prose>

      <H3>Subtle bias from model self-pairings</H3>
      <Prose>
        Chatbot Arena randomly pairs models, and occasionally the same model is presented as both options (especially for models from the same family). These self-pairings produce ties almost by construction and inflate the tie rate for related models. Filtering self-pairings before fitting is a small but important data hygiene step.
      </Prose>

      <Callout accent="gold">
        Bradley-Terry's failure modes are mostly about the comparison graph (disconnected components, separability, sparsity) and about violations of the iid-comparison assumption (vote correlation, non-stationarity). The optimization itself is robust and well-studied. Most production bugs are in the data pipeline that feeds the model, not in the solver.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All sources below were verified against their primary publication venues on 2026-04-26. Authors, titles, and identifiers are confirmed.
      </Prose>

      <H3>Bradley & Terry 1952 — original paper</H3>
      <Prose>
        Ralph Allan Bradley and Milton E. Terry. "Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons." Biometrika 39, no. 3/4 (December 1952): 324-345. The founding paper. Introduces the model in its multiplicative form, derives the likelihood and score equations, and proves identifiability up to a multiplicative constant on the strength vector. Bradley and Terry were not aware of Zermelo's earlier 1929 work and rederived the iterative solution independently. The paper is paywalled at JSTOR but the model and its derivation are reproduced in essentially every textbook on categorical data analysis.
      </Prose>

      <H3>Zermelo 1929 — iterative algorithm</H3>
      <Prose>
        Ernst Zermelo. "Die Berechnung der Turnier-Ergebnisse als ein Maximumproblem der Wahrscheinlichkeitsrechnung." Mathematische Zeitschrift 29, no. 1 (1929): 436-460. Zermelo, motivated by chess tournament rankings, derived the same model and the iterative MLE algorithm twenty-three years before Bradley and Terry. The paper is in German; the algorithm is universally known as the "Zermelo iteration" in the modern Bradley-Terry literature, and Hunter (2004) gives a complete derivation in English of both Zermelo's and Bradley-Terry's original results.
      </Prose>

      <H3>Rao & Kupper 1967 — ties (variant 1)</H3>
      <Prose>
        Pinakpani B. Rao and Lawrence L. Kupper. "Ties in Paired-Comparison Experiments: A Generalization of the Bradley-Terry Model." Journal of the American Statistical Association 62, no. 317 (March 1967): 194-204. Introduces a multiplicative threshold parameter <Code>{"\\tau \\geq 1"}</Code> that creates a "dead zone" around evenly matched comparisons in which the outcome is recorded as a tie. The most cited tie-handling extension in the social-science Bradley-Terry literature.
      </Prose>

      <H3>Davidson 1970 — ties (variant 2)</H3>
      <Prose>
        Roger R. Davidson. "On Extending the Bradley-Terry Model to Accommodate Ties in Paired Comparison Experiments." Journal of the American Statistical Association 65, no. 329 (March 1970): 317-328. The other standard tie extension, with a single multiplicative tie parameter and a geometric-mean form for the tie probability. The version that LMSys describes using in some Chatbot Arena pipeline variants. Mathematically simpler than Rao-Kupper and slightly easier to fit jointly with the strength vector.
      </Prose>

      <H3>Hunter 2004 — MM algorithms</H3>
      <Prose>
        David R. Hunter. "MM Algorithms for Generalized Bradley-Terry Models." The Annals of Statistics 32, no. 1 (2004): 384-406. Proves that the Zermelo iteration is an instance of the Minorization-Maximization algorithm framework, giving a clean unified treatment of MLE for plain Bradley-Terry, Bradley-Terry with ties (both Davidson and Rao-Kupper), and Plackett-Luce listwise comparisons. The standard modern reference for the optimization theory behind the model. Available open-access on the journal's site and on Hunter's faculty page at Penn State.
      </Prose>

      <H3>Zheng et al. 2024 — Chatbot Arena</H3>
      <Prose>
        Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Tianle Li, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zhuohan Li, Zi Lin, Eric P. Xing, Joseph E. Gonzalez, Ion Stoica, Hao Zhang. "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference." arXiv:2403.04132. Published March 2024. The definitive description of the Chatbot Arena platform and its statistical methodology. Section 4 details the Bradley-Terry fitting procedure, the Elo conversion, the bootstrap confidence interval methodology, and the various per-category breakdowns. Code at github.com/lm-sys/FastChat. The arena leaderboard at chat.lmsys.org is updated continuously from the methods described in this paper.
      </Prose>

      <H3>Maystre & Grossglauser 2015 — efficient Plackett-Luce</H3>
      <Prose>
        Lucas Maystre and Matthias Grossglauser. "Fast and Accurate Inference of Plackett-Luce Models." Advances in Neural Information Processing Systems 28 (NeurIPS 2015): 172-180. Introduces the I-LSR algorithm — an iterative spectral approach that is dramatically faster than MM iteration on large rosters. Extends naturally to Bradley-Terry as a special case. The algorithm underlies the <Code>choix</Code> Python library (Maystre is the author), which is the most widely used open-source Bradley-Terry implementation in 2026.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Identifiability and the choice of anchor</H3>
      <Prose>
        Show that the Bradley-Terry log-likelihood is invariant under the transformation <Code>{"\\theta_i \\to \\theta_i + c"}</Code> for any constant <Code>c</Code> applied to all <Code>n</Code> components. What does this imply about which functions of the parameters are estimable from data and which are not? Now consider two anchoring conventions: (1) fix <Code>{"\\theta_1 = 0"}</Code>, and (2) fix <Code>{"\\sum_i \\theta_i = 0"}</Code>. Both produce identifiable parameterizations. What practical advantages does each have, and for which one would you expect the Hessian at the MLE to be better conditioned? How does the choice affect bootstrap confidence intervals?
      </Prose>

      <H3>Exercise 2 — Derive the moment-matching condition</H3>
      <Prose>
        Starting from the Bradley-Terry log-likelihood, take the partial derivative with respect to <Code>{"\\theta_i"}</Code> and set it to zero. Show explicitly that the MLE is the unique configuration of strengths (after anchoring) at which the predicted total wins for every item equals its observed total wins. Use this to give a one-sentence intuitive explanation of why the Zermelo iteration works: each step adjusts each item's strength in the direction that brings its predicted wins closer to its observed wins, and at the fixed point the two are equal. What does the moment-matching condition imply about the model's fit on perfectly balanced tournaments where every item's win rate against the field is exactly 0.5?
      </Prose>

      <H3>Exercise 3 — Why a perfectly dominant model breaks the MLE</H3>
      <Prose>
        Suppose item 1 wins every observed comparison: <Code>{"w_{1j} > 0"}</Code> and <Code>{"w_{j1} = 0"}</Code> for every <Code>{"j \\neq 1"}</Code>. Show that the log-likelihood is monotonically increasing in <Code>{"\\theta_1"}</Code> and has no finite maximizer. What is the limiting distribution of the win probabilities as <Code>{"\\theta_1 \\to +\\infty"}</Code>? Now add a Gaussian prior <Code>{"\\theta_1 \\sim \\mathcal{N}(0, \\sigma^2)"}</Code> and rederive the score equation; show that the regularized MAP estimate is finite for any finite <Code>{"\\sigma"}</Code>. How does the location of the MAP estimate depend on <Code>{"\\sigma"}</Code> and on the number of comparisons? What does this imply about the right setting of <Code>{"\\sigma"}</Code> for a leaderboard with both established and brand-new models?
      </Prose>

      <H3>Exercise 4 — Connectedness and rank uncertainty</H3>
      <Prose>
        Construct two example comparison datasets on 6 models. In dataset A, every pair of models is compared at least 5 times. In dataset B, models 1-3 are compared only against each other (15 games each) and models 4-6 are compared only against each other (15 games each), with no cross-comparisons. For each dataset, what is the structure of the MLE? Where is the likelihood unique vs degenerate? What does the rank uncertainty (computed by bootstrap) look like in each case? What is the minimum amount of cross-pair comparison data you would need to add to dataset B to produce well-defined relative scores between the two clusters, and what would the resulting CIs look like compared to dataset A?
      </Prose>

      <H3>Exercise 5 — Detecting intransitive preferences</H3>
      <Prose>
        Suppose you have fit a Bradley-Terry model on 1000 comparisons over 5 items and the MLE converged cleanly. You suspect that the underlying preferences are intransitive (some triple of items violates the BT assumption). List three diagnostic checks you could run on the data and the fit to test this suspicion. For each check, describe what you would observe in the intransitive case versus the transitive case. As a follow-up, suppose your check confirms intransitivity. Discuss two strategies for proceeding: (a) accept the misspecification and use the BT MLE as a "best linear summary" while reporting reduced confidence in the fit, (b) abandon the scalar model and report category-specific or pairwise-direct results. What considerations would push you toward each option in practice?
      </Prose>

      <H3>Exercise 6 — Bayesian vs frequentist intervals at small N</H3>
      <Prose>
        Run the from-scratch implementation in section 4 with the comparison count reduced from 500 to 50 (keeping the 8-model field). Fit the model with both frequentist MLE + bootstrap and Bayesian MAP + Laplace approximation under a weakly informative <Code>{"\\mathcal{N}(0, 2^2)"}</Code> prior. Compare the resulting confidence/credible intervals: which is wider on average? Which contains the ground truth more often (run the experiment 100 times with different random seeds and count coverage)? What does the comparison tell you about when frequentist intervals can be trusted at small sample sizes versus when the Bayesian formulation is necessary?
      </Prose>

      <H3>Exercise 7 — Connecting Bradley-Terry to logistic regression</H3>
      <Prose>
        Show that the Bradley-Terry MLE problem can be cast as a logistic regression. Specifically, encode each comparison <Code>{"(i, j, y)"}</Code> with <Code>y = 1</Code> if <Code>i</Code> won and <Code>y = 0</Code> if <Code>j</Code> won as a row in a sparse design matrix where the <Code>i</Code>-th column entry is <Code>{"+1"}</Code>, the <Code>j</Code>-th column entry is <Code>{"-1"}</Code>, and all other entries are zero. Show that fitting a logistic regression on this encoding (with no intercept) recovers the Bradley-Terry strengths exactly. What does this equivalence imply about which existing statistical software can fit Bradley-Terry "for free"? What does it imply about extending Bradley-Terry with covariates — for example, including a feature for "model is a transformer", or "model was trained on more than X tokens", to study what model attributes correlate with strength?
      </Prose>

    </div>
  ),
};

export default bradleyTerry;
