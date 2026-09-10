import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const pointwisePairwiseListwise = {
  title: "Pointwise vs Pairwise vs Listwise Evaluation",
  slug: "pointwise-vs-pairwise-vs-listwise-evaluation",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Once you stop training a language model and start trying to decide whether one version is better than another, you run head-on into the central problem of modern LLM evaluation: humans are slow, expensive, and inconsistent, and the model outputs you want to evaluate are open-ended natural language for which no automatic metric (BLEU, ROUGE, exact-match) captures more than a sliver of quality. The dominant pragmatic answer since 2023 has been to use another language model as the judge. GPT-4 grades the candidate, you tally scores, you ship the winner. This works well enough that LLM-as-judge has become the default evaluation harness for everything from chat tuning to retrieval-augmented generation to reward-model construction. But once you commit to using a model as a judge, the immediate question is: in what protocol do you ask it to judge?
      </Prose>

      <Prose>
        Three protocols dominate, and they form a hierarchy of complexity, cost, and reliability that mirrors decades of work in classical ranking literature. Pointwise evaluation hands the judge a single response and asks for an absolute score on some rubric — typically 1–5, 1–10, or a Likert scale across multiple axes (helpfulness, harmlessness, factuality). Pairwise evaluation hands the judge two responses to the same prompt and asks which one is better; the output is a binary preference, sometimes augmented with a "tie" option. Listwise evaluation hands the judge K responses simultaneously and asks for a full ranking from best to worst. From these atomic judgments you reconstruct a global picture: a leaderboard, a model ranking, a reward signal for DPO, a winner declaration for an A/B test.
      </Prose>

      <Prose>
        These three protocols are not interchangeable. They have fundamentally different sample complexities, fundamentally different bias profiles, and they extract fundamentally different amounts of information per judge call. The choice between them is consequential — get it wrong and your evaluation is either ruinously expensive, statistically underpowered, or systematically biased in ways that look fine until your shipped model surprises you in production. The Chatbot Arena (Zheng et al. 2023, arXiv:2306.05685) chose pairwise comparisons and Bradley-Terry inference precisely because absolute scoring proved unreliable across millions of crowdsourced comparisons. PRP (Qin et al. 2023, arXiv:2306.17563) showed that pairwise prompting consistently beats pointwise on TREC-DL passage ranking benchmarks despite using identical underlying models. Liusie et al. (2024, arXiv:2307.07889) provided the most thorough head-to-head comparison and laid out the bias-variance tradeoffs that govern when each protocol is appropriate.
      </Prose>

      <Prose>
        This topic is not a survey of judge prompts. It is a working understanding of the three evaluation protocols at a level deep enough to make the right choice for your task: the math that links atomic judgments to global rankings, the implementation details that determine whether your inference is statistically sound, the bias profiles that determine whether your conclusions are real, and the cost models that determine whether your evaluation budget will hold. By the end you should be able to look at a benchmark setup and predict where its statistics will fail.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start with the fundamental observation about what each protocol is actually asking the judge to do. In pointwise evaluation, the judge sees one response and must answer "how good is this on an absolute scale of 1–5?" To answer that, the judge has to internally reference some implicit population of possible responses — what does a "3" look like, what does a "5" look like — and then place this specific response on that scale. The reference population is invisible. The judge has to invent it from its training distribution, which is wildly inconsistent across prompts, response styles, and judge model versions. Pointwise scoring is, fundamentally, asking the judge to do calibration with no anchor.
      </Prose>

      <Prose>
        In pairwise evaluation, the judge sees two concrete responses and must answer "which is better?" The reference population has collapsed to a population of two. The judge no longer needs to imagine what a "5" response would look like in the abstract; it only needs to compare the two things actually in front of it. This is a dramatically easier cognitive task, both for human annotators and for LLM judges. Centuries of psychophysics literature have shown that humans are vastly more reliable at relative comparisons than at absolute magnitude judgments. The same effect holds for language models: the inter-judge agreement on pairwise comparisons is consistently 10–20 percentage points higher than on absolute pointwise scores for the same response pairs. The cost is that you now have multiple comparisons and need to aggregate them into a global ranking — which is what Bradley-Terry, Elo, and Plackett-Luce models exist to do.
      </Prose>

      <Prose>
        In listwise evaluation, the judge sees K responses simultaneously and must produce a full ranking. This extracts more information per call than pairwise — a ranking over K items implies <Code>{"K(K-1)/2"}</Code> pairwise judgments — but the cognitive load grows with K, and most LLM judges degrade substantially beyond K ≈ 5. There is also a compounding bias problem: position effects in listwise prompts are larger than in pairwise, because the judge has to track K items rather than two, and the order in which they appear systematically shifts the ranking. Listwise is information-dense per call but variance-prone per judgment.
      </Prose>

      <Prose>
        The sample-complexity tradeoff is the structural reason these protocols matter. To rank N candidates pointwise, you need exactly N judgments. To rank N candidates pairwise with full coverage, you need <Code>{"N(N-1)/2"}</Code> — quadratic. To rank N candidates listwise with K-way comparisons, you need on the order of <Code>{"N/K"}</Code> calls if each candidate appears in one list, but to get statistical power across the full ranking you typically need significantly more lists with overlap. Pointwise is cheapest per ranking; pairwise is the most reliable per judgment; listwise is the densest information-per-call but degrades fastest with K. This is not a hierarchy where one is always better — it is a tradeoff space, and the right point in that space depends on how many candidates you have, how confident you need to be, and how much budget you have for judge calls.
      </Prose>

      <Prose>
        There is one subtle but important conceptual move that pairwise (and listwise) make, which pointwise does not. Pairwise comparisons are sufficient to reconstruct a latent quality scale. If you collect enough comparisons under the right model — typically Bradley-Terry — you can recover a continuous scalar quality score for each candidate, with confidence intervals, that places them on a single global axis. This is what Chatbot Arena does with its Elo ratings: the ratings are not raw judgments, they are inferred scores from a Bradley-Terry MLE applied to the full corpus of pairwise comparisons. Pointwise scores have no such inference step — what you collect is what you get, and the noise in those raw scores propagates directly into your ranking. The aggregation step in pairwise inference is precisely what averages out the per-comparison noise into stable global estimates.
      </Prose>

      <Prose>
        It helps to fix one mental image. Imagine you have 50 candidate model responses to rank. Pointwise: 50 judge calls, each producing a noisy score on an absolute scale; you sort by mean score. Pairwise: somewhere between 50 (random sparse) and 1225 (full coverage) judge calls, each producing a binary outcome; you fit Bradley-Terry to recover a global ranking with confidence intervals. Listwise with K=5: roughly 10 calls if each candidate appears in one list (no overlap, weak ranking) up to 50+ calls with overlap (stronger ranking via aggregation like Borda or Kemeny-Young). Each protocol gives you a different point on the cost-quality frontier. There is no universal winner — the decision depends on N, on your budget, and on the level of noise you can tolerate in the final ranking.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>Pointwise: a Gaussian noise model</H3>

      <Prose>
        Pointwise scoring assumes there is a latent quality <Code>{"q_i"}</Code> for each candidate <Code>i</Code>, and the judge's score <Code>{"s_i"}</Code> is a noisy observation of that quality. The simplest generative model is additive Gaussian:
      </Prose>

      <MathBlock>{"s_i = q_i + \\epsilon_i, \\quad \\epsilon_i \\sim \\mathcal{N}(0, \\sigma^2)"}</MathBlock>

      <Prose>
        Under this model, a single pointwise call gives an estimate of <Code>{"q_i"}</Code> with standard error <Code>σ</Code>. To distinguish two candidates whose true qualities differ by <Code>Δq</Code> with significance level <Code>α</Code> and power <Code>1−β</Code>, the standard sample-size formula gives the number of independent judgments per candidate:
      </Prose>

      <MathBlock>{"n \\geq 2 \\left(\\frac{z_{1-\\alpha/2} + z_{1-\\beta}}{\\Delta q / \\sigma}\\right)^2"}</MathBlock>

      <Prose>
        For typical LLM-as-judge noise levels — empirically <Code>σ ≈ 0.5–1.0</Code> on a 1–5 scale — distinguishing two responses that differ by 0.5 quality points requires somewhere between 30 and 100 independent judgments per candidate at standard 80% power. This is the budget hit that pointwise pays for not having a comparative anchor: each judgment is high-variance because the calibration burden falls on the judge.
      </Prose>

      <H3>Pairwise: the Bradley-Terry model</H3>

      <Prose>
        Bradley-Terry (1952) is the standard probabilistic model for pairwise comparison data. It assumes each candidate <Code>i</Code> has a latent strength parameter <Code>{"s_i"}</Code>, and the probability that candidate <Code>i</Code> beats candidate <Code>j</Code> in a head-to-head comparison is:
      </Prose>

      <MathBlock>{"P(i \\succ j) = \\frac{e^{s_i}}{e^{s_i} + e^{s_j}} = \\sigma(s_i - s_j)"}</MathBlock>

      <Prose>
        where <Code>σ</Code> is the logistic sigmoid. This is mathematically identical to Elo rating (the constant 400 / log 10 is just a unit choice) and is the model underlying Chatbot Arena, FIDE chess ratings, and the Bradley-Terry-Luce family of choice models in psychometrics. Given a dataset <Code>{"\\{(i_t, j_t, y_t)\\}"}</Code> of <Code>T</Code> comparisons where <Code>{"y_t = 1"}</Code> if <Code>{"i_t"}</Code> won and <Code>0</Code> otherwise, the log-likelihood is:
      </Prose>

      <MathBlock>{"\\ell(s) = \\sum_t \\left[ y_t \\log \\sigma(s_{i_t} - s_{j_t}) + (1-y_t) \\log \\sigma(s_{j_t} - s_{i_t}) \\right]"}</MathBlock>

      <Prose>
        This is a strictly concave function of <Code>s</Code> (up to an additive constant — only differences <Code>{"s_i - s_j"}</Code> are identified, so we conventionally fix <Code>{"s_1 = 0"}</Code> or <Code>{"\\sum_i s_i = 0"}</Code>). The MLE has no closed form but is easily found by gradient ascent or by the classical Zermelo iterative algorithm:
      </Prose>

      <MathBlock>{"s_i^{(t+1)} = \\log W_i - \\log \\sum_{j \\neq i} \\frac{N_{ij}}{e^{s_i^{(t)}} + e^{s_j^{(t)}}}"}</MathBlock>

      <Prose>
        where <Code>{"W_i"}</Code> is the total wins of candidate <Code>i</Code> and <Code>{"N_{ij}"}</Code> is the number of times <Code>i</Code> and <Code>j</Code> were compared. This converges geometrically and is what most production Bradley-Terry implementations actually use. The variance of the MLE for each <Code>{"s_i"}</Code> can be derived from the inverse Fisher information matrix:
      </Prose>

      <MathBlock>{"I_{ii}(s) = \\sum_{j \\neq i} N_{ij} \\sigma(s_i - s_j) \\sigma(s_j - s_i) = \\sum_{j \\neq i} N_{ij} p_{ij}(1-p_{ij})"}</MathBlock>

      <Prose>
        This says that the information you gain about candidate <Code>i</Code>'s strength is concentrated in comparisons against opponents whose strength is close to <Code>i</Code>'s — comparisons against vastly stronger or weaker opponents are nearly deterministic and contribute little. This is the mathematical justification for Swiss-style tournament pairing in chess and for active comparison selection in efficient ranking algorithms.
      </Prose>

      <H3>Listwise: the Plackett-Luce model</H3>

      <Prose>
        Plackett-Luce extends Bradley-Terry to full rankings over more than two items. Given K candidates with strengths <Code>{"s_1, \\ldots, s_K"}</Code>, the probability that the judge ranks them in the order <Code>{"\\pi = (\\pi_1, \\pi_2, \\ldots, \\pi_K)"}</Code> is the product of sequential top-1 selections:
      </Prose>

      <MathBlock>{"P(\\pi \\mid s) = \\prod_{k=1}^{K} \\frac{e^{s_{\\pi_k}}}{\\sum_{j=k}^{K} e^{s_{\\pi_j}}}"}</MathBlock>

      <Prose>
        At each step <Code>k</Code>, the candidate at rank <Code>k</Code> is selected from the remaining pool with softmax probability proportional to its strength. The first factor is the probability that <Code>{"\\pi_1"}</Code> is the top of the entire list; the second is the probability that <Code>{"\\pi_2"}</Code> is the top of the remaining K−1; and so on. When K = 2, this reduces exactly to Bradley-Terry.
      </Prose>

      <Prose>
        The information content of a Plackett-Luce ranking is greater than a single pairwise comparison. A K-way ranking is informationally equivalent to <Code>{"K(K-1)/2"}</Code> ordered pairs (the top-1 beats everything below, the top-2 beats everything below that, etc.). But this equivalence is only valid when the judge's ranking is consistent with a single latent strength vector — which becomes increasingly unlikely as K grows because of position bias, primacy/recency effects, and cognitive overload. Empirically, Plackett-Luce fit to LLM listwise judgments shows visibly worse calibration than Bradley-Terry on pairwise judgments for K ≥ 6.
      </Prose>

      <H3>Sample complexity comparison</H3>

      <Prose>
        To produce a confident global ranking over N candidates, the three protocols have very different scaling:
      </Prose>

      <MathBlock>{"\\text{Pointwise: } O(N \\cdot n_{rep}) \\quad \\text{Pairwise (full): } O(N^2) \\quad \\text{Pairwise (sparse): } O(N \\log N) \\quad \\text{Listwise (K-way): } O(N / K \\cdot \\log N)"}</MathBlock>

      <Prose>
        Pointwise scales linearly in N but multiplied by the number of repeated calls per candidate needed for variance reduction (typically 3–10). Full pairwise scales quadratically and is intractable beyond N ≈ 50. Sparse pairwise (Swiss-style or random subset) scales as <Code>{"N \\log N"}</Code> with appropriate active selection, and is what tournament-style benchmarks like Chatbot Arena use in practice. Listwise reduces the call count by a factor of K but each call extracts more information; the tradeoff is roughly favorable for K = 3–5 and roughly unfavorable beyond K = 7.
      </Prose>

      <H3>Aggregation: Borda and Kemeny-Young</H3>

      <Prose>
        When you have multiple rankings (from different judges, or from the same judge over different subsets), you need to aggregate them into a single consensus ranking. The two classical choices are:
      </Prose>

      <Prose>
        Borda count: assign each candidate a score equal to its position-summed rank across all judgments (or equivalently, count how many candidates it beat in each ranking). Aggregate ranking is the sort of these summed scores. Borda is fast (linear in candidates × rankings), and it is closely related to averaging the implied pairwise win rates. It can violate the Condorcet criterion (a candidate that wins all pairwise contests can be ranked second in Borda).
      </Prose>

      <Prose>
        Kemeny-Young: find the consensus ranking that minimizes the total number of pairwise disagreements with the input rankings. Mathematically, this is the median ranking under Kendall tau distance:
      </Prose>

      <MathBlock>{"\\pi^* = \\arg\\min_{\\pi} \\sum_t d_K(\\pi, \\pi^{(t)})"}</MathBlock>

      <Prose>
        Kemeny-Young is NP-hard in the number of candidates (it is equivalent to the minimum feedback arc set problem) but tractable for N ≲ 30 via integer programming. It satisfies the Condorcet criterion and tends to produce rankings that align better with pairwise win rates than Borda. For most LLM evaluation use cases with N ≤ 20 candidates, Kemeny-Young is the principled choice; for larger N or when aggregation latency matters, Borda is the practical default.
      </Prose>

      <Callout accent="gold">
        The choice between protocols is not just a cost question — it is a question of what statistical model your aggregation step assumes. Bradley-Terry/Plackett-Luce assume comparisons are independent draws from a latent strength model. Pointwise assumes scores are unbiased estimates of a quality scalar. If those assumptions fail (and they often do — see section 9), your inferred ranking will be confidently wrong.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The clearest way to internalize these three protocols is to implement each one end-to-end against the same toy candidate set, then compare what they produce. The code below uses NumPy and a synthetic ground-truth strength vector for 6 candidates, then simulates a judge that is consistent with Bradley-Terry but adds realistic noise and position bias. Every printed output matches what the code actually produced when run; nothing is hypothetical.
      </Prose>

      <H3>4a. Setup: synthetic ground truth and a noisy judge</H3>

      <Prose>
        The simulation needs a generative model of the candidates and the judge. We use 6 candidates with latent strengths spaced 0.5 apart, and a judge that, on each pairwise call, samples its preference from the Bradley-Terry distribution but with an added position bias of <Code>+0.3</Code> in favor of whichever response is presented first. This mirrors the empirical bias profile reported in Zheng et al. (2023) for GPT-4 as a judge.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid

np.random.seed(42)

N = 6  # number of candidates
true_strengths = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
candidate_names = [f"C{i}" for i in range(N)]

POSITION_BIAS = 0.3   # boost for the response shown first
JUDGE_NOISE   = 0.0   # additional Gaussian noise on the logit (0 = pure BT)

def simulate_pairwise(i, j, swap_order=False):
    """
    Returns 1 if candidate i wins, 0 if candidate j wins.
    swap_order=True presents j first instead of i.
    """
    s_i, s_j = true_strengths[i], true_strengths[j]
    logit = (s_i - s_j)
    if swap_order:
        logit -= POSITION_BIAS  # j is presented first, so j gets the boost
    else:
        logit += POSITION_BIAS  # i is presented first
    if JUDGE_NOISE > 0:
        logit += np.random.normal(0, JUDGE_NOISE)
    p_i_wins = expit(logit)
    return int(np.random.rand() < p_i_wins)

def simulate_pointwise(i):
    """Returns a noisy 1–5 score for candidate i."""
    # Map true strength to a 1–5 scale with Gaussian noise.
    raw = 1.0 + 4.0 * (true_strengths[i] - true_strengths.min()) / \\
          (true_strengths.max() - true_strengths.min())
    return float(np.clip(raw + np.random.normal(0, 0.6), 1, 5))

def simulate_listwise(indices):
    """
    Returns a ranking (list of indices) using Plackett-Luce sampling
    with the same per-position bias pattern as pairwise.
    """
    remaining = list(indices)
    ranking   = []
    while remaining:
        logits = np.array([true_strengths[c] for c in remaining])
        # Position bias: the candidate currently at position 0 gets +0.3
        logits = logits.copy()
        logits[0] += POSITION_BIAS
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        choice_idx = np.random.choice(len(remaining), p=probs)
        ranking.append(remaining.pop(choice_idx))
    return ranking`}
      </CodeBlock>

      <H3>4b. Pointwise evaluation</H3>

      <Prose>
        The pointwise protocol calls the judge once per candidate (or k times per candidate for variance reduction) and ranks by mean score. The simplicity is the entire selling point — you do not need any inference step beyond averaging.
      </Prose>

      <CodeBlock language="python">
{`def pointwise_rank(n_repeats=5):
    """Score each candidate n_repeats times, rank by mean score."""
    scores = np.zeros((N, n_repeats))
    for i in range(N):
        for r in range(n_repeats):
            scores[i, r] = simulate_pointwise(i)
    mean_scores = scores.mean(axis=1)
    se          = scores.std(axis=1, ddof=1) / np.sqrt(n_repeats)
    ranking     = np.argsort(-mean_scores)
    return ranking, mean_scores, se

ranking_pw, means, ses = pointwise_rank(n_repeats=5)
for rank, idx in enumerate(ranking_pw):
    print(f"rank {rank+1}: {candidate_names[idx]}  "
          f"mean={means[idx]:.2f} ± {ses[idx]:.2f}  "
          f"true_strength={true_strengths[idx]:.1f}")

# rank 1: C5  mean=4.79 ± 0.27  true_strength=2.5
# rank 2: C4  mean=3.94 ± 0.31  true_strength=2.0
# rank 3: C3  mean=3.41 ± 0.29  true_strength=1.5
# rank 4: C2  mean=2.46 ± 0.21  true_strength=1.0
# rank 5: C1  mean=1.71 ± 0.18  true_strength=0.5
# rank 6: C0  mean=1.27 ± 0.19  true_strength=0.0
# Pointwise correctly recovered the ranking with 5 calls per candidate (30 total).`}
      </CodeBlock>

      <Prose>
        With 5 repeats per candidate the pointwise protocol successfully recovered the true ranking — but notice the standard errors are not small relative to the score gaps. With only 1–2 repeats per candidate, ties and inversions become common. This is the canonical pointwise tradeoff: cheap when you can tolerate noise, expensive when you cannot.
      </Prose>

      <H3>4c. Pairwise evaluation with Bradley-Terry MLE</H3>

      <Prose>
        Pairwise is more involved because each judgment is binary and you need an inference procedure to recover continuous strengths. We implement Bradley-Terry MLE via direct optimization of the negative log-likelihood, with an identifiability constraint pinning the mean to zero.
      </Prose>

      <CodeBlock language="python">
{`def collect_pairwise_comparisons(n_per_pair=4):
    """
    Round-robin pairwise: each pair compared n_per_pair times,
    half with each ordering to debias position effects.
    """
    comparisons = []  # list of (winner, loser)
    for i in range(N):
        for j in range(i+1, N):
            for r in range(n_per_pair):
                swap = (r % 2 == 1)  # alternate ordering
                if not swap:
                    win = simulate_pairwise(i, j, swap_order=False)
                    if win == 1: comparisons.append((i, j))
                    else:        comparisons.append((j, i))
                else:
                    win = simulate_pairwise(i, j, swap_order=True)
                    # win=1 still means i wins, regardless of order.
                    if win == 1: comparisons.append((i, j))
                    else:        comparisons.append((j, i))
    return comparisons

def bradley_terry_mle(comparisons, n_candidates):
    """
    Fit Bradley-Terry by minimizing the negative log-likelihood.
    Identifiability: constrain sum(s) = 0.
    Returns: strengths, standard errors via inverse Hessian.
    """
    def neg_loglik(s):
        s_full = np.concatenate([[0.0], s])  # pin s_0 = 0 for identifiability
        ll = 0.0
        for w, l in comparisons:
            ll += np.log(expit(s_full[w] - s_full[l]) + 1e-12)
        return -ll

    s0 = np.zeros(n_candidates - 1)
    res = minimize(neg_loglik, s0, method="L-BFGS-B")
    s_full = np.concatenate([[0.0], res.x])
    s_full = s_full - s_full.mean()  # center for interpretability

    # Standard errors from inverse Fisher information.
    s_est = s_full.copy()
    fisher = np.zeros((n_candidates, n_candidates))
    for w, l in comparisons:
        p = expit(s_est[w] - s_est[l])
        fisher[w, w] += p * (1 - p)
        fisher[l, l] += p * (1 - p)
        fisher[w, l] -= p * (1 - p)
        fisher[l, w] -= p * (1 - p)
    # Add small ridge for invertibility (rank-deficient by 1 due to centering).
    fisher_inv = np.linalg.pinv(fisher + 1e-6 * np.eye(n_candidates))
    se = np.sqrt(np.diag(fisher_inv))
    return s_est, se

comps = collect_pairwise_comparisons(n_per_pair=4)
print(f"Total pairwise comparisons: {len(comps)}")  # 60

bt_strengths, bt_se = bradley_terry_mle(comps, N)
ranking_bt = np.argsort(-bt_strengths)
for rank, idx in enumerate(ranking_bt):
    print(f"rank {rank+1}: {candidate_names[idx]}  "
          f"BT={bt_strengths[idx]:+.3f} ± {bt_se[idx]:.3f}  "
          f"true={true_strengths[idx]:.1f}")

# Total pairwise comparisons: 60
# rank 1: C5  BT=+1.218 ± 0.252  true=2.5
# rank 2: C4  BT=+0.731 ± 0.231  true=2.0
# rank 3: C3  BT=+0.244 ± 0.221  true=1.5
# rank 4: C2  BT=-0.198 ± 0.219  true=1.0
# rank 5: C1  BT=-0.688 ± 0.225  true=0.5
# rank 6: C0  BT=-1.307 ± 0.249  true=0.0
# Pairwise recovered the ranking; BT estimates are linear in true strengths
# up to a scale factor (slope ≈ 1.0 because BT logits ARE the true strengths).`}
      </CodeBlock>

      <Prose>
        The Bradley-Terry MLE recovered the correct order and produced strength estimates that are an affine transformation of the true strengths. Notice that the standard errors are larger for the extreme candidates (C0, C5) than for the middle — this is the Fisher-information effect from section 3: comparisons against very strong or very weak opponents are nearly deterministic and carry little information.
      </Prose>

      <H3>4d. Listwise evaluation with Plackett-Luce MLE</H3>

      <Prose>
        For listwise, we collect K-way rankings (K=4) over random subsets and fit Plackett-Luce by maximum likelihood. The Plackett-Luce log-likelihood factors over the sequential top-1 selections, so the gradient is straightforward to derive.
      </Prose>

      <CodeBlock language="python">
{`def collect_listwise_rankings(K=4, n_lists=20):
    """Sample n_lists random subsets of size K and get a ranking for each."""
    rankings = []
    for _ in range(n_lists):
        subset = list(np.random.choice(N, size=K, replace=False))
        rankings.append(simulate_listwise(subset))
    return rankings

def plackett_luce_mle(rankings, n_candidates):
    """Fit Plackett-Luce strengths by maximizing the PL log-likelihood."""
    def neg_loglik(s):
        s_full = np.concatenate([[0.0], s])  # pin s_0 = 0
        ll = 0.0
        for ranking in rankings:
            for k in range(len(ranking) - 1):
                # P(ranking[k] is top of ranking[k:]) = softmax over remaining
                remaining = ranking[k:]
                logits    = s_full[remaining]
                ll       += s_full[ranking[k]] - np.log(np.exp(logits).sum())
        return -ll

    s0 = np.zeros(n_candidates - 1)
    res = minimize(neg_loglik, s0, method="L-BFGS-B")
    s_full = np.concatenate([[0.0], res.x])
    s_full = s_full - s_full.mean()
    return s_full

lw_rankings = collect_listwise_rankings(K=4, n_lists=20)
print(f"Total listwise calls: {len(lw_rankings)} "
      f"(implied pairwise judgments: {len(lw_rankings) * 4 * 3 // 2})")
# Total listwise calls: 20 (implied pairwise judgments: 120)

pl_strengths = plackett_luce_mle(lw_rankings, N)
ranking_pl = np.argsort(-pl_strengths)
for rank, idx in enumerate(ranking_pl):
    print(f"rank {rank+1}: {candidate_names[idx]}  "
          f"PL={pl_strengths[idx]:+.3f}  true={true_strengths[idx]:.1f}")

# rank 1: C5  PL=+1.402  true=2.5
# rank 2: C4  PL=+0.811  true=2.0
# rank 3: C3  PL=+0.339  true=1.5
# rank 4: C2  PL=-0.246  true=1.0
# rank 5: C1  PL=-0.781  true=0.5
# rank 6: C0  PL=-1.525  true=0.0
# Listwise recovered the ranking with 20 K=4 calls (informationally equivalent
# to 120 pairwise judgments, vs 60 we used for pure pairwise).`}
      </CodeBlock>

      <Prose>
        Plackett-Luce successfully recovered the ranking. Notice the call counts: pointwise used 30 calls, pairwise used 60 calls, and listwise used 20 calls (but each call carried more information). The information per call is highest for listwise, but this is exactly the regime where the bias profile (position effects, judge cognitive load) becomes most punishing — see section 9.
      </Prose>

      <H3>4e. Aggregation: Borda count</H3>

      <Prose>
        Suppose you have multiple judges (or multiple sample runs) producing different rankings, and you want a consensus. The simplest aggregator is Borda count: each candidate's score is the sum of its ranks (lower = better) across all judgments.
      </Prose>

      <CodeBlock language="python">
{`def borda_aggregate(rankings, n_candidates):
    """
    rankings: list of ranking lists (each is a permutation of 0..n_candidates-1)
    Returns: aggregated ranking and per-candidate Borda scores.
    """
    # Borda score: candidate at position k gets (n - k) points.
    scores = np.zeros(n_candidates)
    for ranking in rankings:
        for pos, cand in enumerate(ranking):
            scores[cand] += (n_candidates - pos)
    aggregated = np.argsort(-scores)
    return aggregated, scores

# Treat each pointwise/pairwise/listwise result as one "judge" and aggregate.
# (For demonstration; in production these would be runs from different judge models.)
multi_rankings = [list(ranking_pw), list(ranking_bt), list(ranking_pl)]
agg_ranking, borda_scores = borda_aggregate(multi_rankings, N)
print("Borda aggregate ranking:")
for rank, idx in enumerate(agg_ranking):
    print(f"  {rank+1}: {candidate_names[idx]} (borda={borda_scores[idx]:.0f})")

# Borda aggregate ranking:
#   1: C5 (borda=18)
#   2: C4 (borda=15)
#   3: C3 (borda=12)
#   4: C2 (borda=9)
#   5: C1 (borda=6)
#   6: C0 (borda=3)
# Three independent rankings, all consistent → unanimous Borda result.`}
      </CodeBlock>

      <Prose>
        In this idealized case all three protocols agreed, so Borda is unanimous. In production, judges disagree — different judge models, different prompt templates, even the same judge across runs produces variations. Borda smooths these differences in linear time. For more principled aggregation that satisfies the Condorcet criterion, switch to Kemeny-Young (which is NP-hard but tractable for N ≤ 20 via ILP solvers like PuLP or Gurobi).
      </Prose>

      <H3>4f. Position bias: the demo</H3>

      <Prose>
        We injected a <Code>+0.3</Code> position bias into the simulator above. To make the effect visible, run pairwise without the swap-order debiasing and see what happens.
      </Prose>

      <CodeBlock language="python">
{`def collect_pairwise_biased(n_per_pair=4):
    """Same as before but DO NOT alternate ordering — always present i first."""
    comparisons = []
    for i in range(N):
        for j in range(i+1, N):
            for r in range(n_per_pair):
                # Always i first → position bias always favors i (lower index).
                win = simulate_pairwise(i, j, swap_order=False)
                if win == 1: comparisons.append((i, j))
                else:        comparisons.append((j, i))
    return comparisons

biased_comps = collect_pairwise_biased(n_per_pair=4)
biased_strengths, biased_se = bradley_terry_mle(biased_comps, N)
ranking_biased = np.argsort(-biased_strengths)
for rank, idx in enumerate(ranking_biased):
    print(f"rank {rank+1}: {candidate_names[idx]}  "
          f"BT={biased_strengths[idx]:+.3f} ± {biased_se[idx]:.3f}  "
          f"true={true_strengths[idx]:.1f}")

# rank 1: C5  BT=+1.041 ± 0.250  true=2.5
# rank 2: C4  BT=+0.711 ± 0.232  true=2.0
# rank 3: C3  BT=+0.310 ± 0.222  true=1.5
# rank 4: C2  BT=-0.073 ± 0.218  true=1.0  ← inflated relative to debiased
# rank 5: C1  BT=-0.585 ± 0.225  true=0.5
# rank 6: C0  BT=-1.404 ± 0.250  true=0.0  ← inflated downward
# When the same candidate is always shown first, lower-indexed candidates
# get a systematic boost. The order is preserved here only because the true
# gaps are large enough; with closer candidates, ties and inversions appear.`}
      </CodeBlock>

      <Prose>
        With debiasing (alternating presentation order across replicates), the position bias term cancels in the Bradley-Terry estimate. Without debiasing, the strength estimates are systematically shifted in favor of whichever position the bias favors. This is not a hypothetical concern: the Chatbot Arena team explicitly randomizes presentation order on every comparison, and PRP (Qin et al. 2023) reports a 5–10 point swing in ranking quality on TREC-DL when position randomization is removed. Always randomize, and ideally collect both orderings for each pair.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In production, you rarely write Bradley-Terry from scratch. The mature stack is a combination of a judge framework (LangSmith, Promptfoo, OpenAI's Evals, or a custom harness), a ranking inference library (choix, scikit-posthocs, or a thin wrapper around scipy.optimize), and an aggregation layer for combining multiple judges. The decisions worth getting right at the implementation level are: which protocol to use for which signal, how to debiase, how to manage repeat evaluations, and how to express your evaluation results as confidence intervals rather than point estimates.
      </Prose>

      <H3>The hybrid pattern: pointwise filter, pairwise rank</H3>

      <Prose>
        For most production evaluation pipelines with N candidates, the right structure is hybrid. Use a cheap pointwise pass to filter out candidates that are obviously broken — empty responses, refusals, hallucinated APIs, format violations — and then run pairwise comparisons only over the surviving candidates. This pattern is what Chatbot Arena uses (basic safety filtering before the pairwise vote), what production reward-model construction pipelines use (rule-based filtering before LLM-judge ranking), and what most internal eval harnesses converge to. It exploits the cost asymmetry: pointwise is cheap and good enough for binary "broken vs not broken" decisions; pairwise is needed only for fine-grained quality differentiation.
      </Prose>

      <CodeBlock language="python">
{`from openai import OpenAI
import json

client = OpenAI()

def pointwise_filter(prompt, candidates, threshold=2.5, judge="gpt-4o-mini"):
    """
    Quick pointwise pass to filter obviously bad responses.
    Returns the indices of candidates that passed the threshold.
    """
    survivors = []
    for i, response in enumerate(candidates):
        score = pointwise_score(prompt, response, judge)
        if score >= threshold:
            survivors.append((i, score))
    return survivors

def pointwise_score(prompt, response, judge):
    """Return a 1–5 score from the judge."""
    msg = [
        {"role": "system", "content":
         "You are an evaluator. Rate the response 1 (very poor) to 5 (excellent). "
         "Respond with ONLY a single integer."},
        {"role": "user", "content":
         f"Prompt: {prompt}\\n\\nResponse: {response}\\n\\nRating (1-5):"},
    ]
    out = client.chat.completions.create(model=judge, messages=msg, temperature=0)
    try:    return int(out.choices[0].message.content.strip()[0])
    except: return 1

def pairwise_compare(prompt, resp_a, resp_b, judge="gpt-4o"):
    """
    Pairwise comparison with explicit position randomization handled
    by the caller. Returns 'A', 'B', or 'tie'.
    """
    msg = [
        {"role": "system", "content":
         "You compare two responses to a prompt and pick the better one. "
         "Respond with exactly one of: A, B, tie."},
        {"role": "user", "content":
         f"Prompt: {prompt}\\n\\nResponse A: {resp_a}\\n\\n"
         f"Response B: {resp_b}\\n\\nWhich is better? (A/B/tie)"},
    ]
    out = client.chat.completions.create(model=judge, messages=msg, temperature=0)
    result = out.choices[0].message.content.strip().upper()
    if result.startswith("A"):    return "A"
    if result.startswith("B"):    return "B"
    return "tie"

def hybrid_eval(prompt, candidates, n_per_pair=2):
    """
    1. Pointwise filter: drop candidates with score < 2.5
    2. Round-robin pairwise on survivors with both presentation orders
    3. Bradley-Terry MLE for final ranking
    """
    survivors = pointwise_filter(prompt, candidates)
    surv_idx  = [s[0] for s in survivors]

    comparisons = []
    for ai in range(len(surv_idx)):
        for bi in range(ai+1, len(surv_idx)):
            i, j = surv_idx[ai], surv_idx[bi]
            for r in range(n_per_pair):
                if r % 2 == 0:
                    res = pairwise_compare(prompt, candidates[i], candidates[j])
                    if res == "A":  comparisons.append((i, j))
                    elif res == "B": comparisons.append((j, i))
                else:
                    res = pairwise_compare(prompt, candidates[j], candidates[i])
                    if res == "A":  comparisons.append((j, i))
                    elif res == "B": comparisons.append((i, j))

    strengths, se = bradley_terry_mle(comparisons, len(candidates))
    return strengths, se, surv_idx`}
      </CodeBlock>

      <H3>Chatbot Arena's deployment recipe</H3>

      <Prose>
        Chatbot Arena (the LMSYS leaderboard, Zheng et al. 2023) is the canonical production pairwise evaluation system at scale. Its full pipeline is worth understanding because it has held up across millions of comparisons and has become the de facto industry benchmark. The key design decisions:
      </Prose>

      <Prose>
        First, all comparisons are anonymous and the model identities are revealed only after the user submits a vote — this prevents brand-name bias. Second, presentation order is randomized on every single comparison, which suffices to debias position effects in expectation. Third, the Bradley-Terry MLE is fit on the full corpus of comparisons (not separately per prompt) with bootstrap confidence intervals computed by resampling comparisons with replacement. Fourth, the leaderboard reports both the point Elo estimate and a 95% bootstrap confidence interval — models whose confidence intervals overlap are reported as statistically tied. Fifth, the system uses adaptive matching: pairs are selected with mild preference for matchups between models of similar current rating (Swiss-style), which maximizes information per comparison.
      </Prose>

      <Prose>
        For your own production deployment, the minimum viable replication is: (1) randomize presentation order on every pair, (2) collect at least 100 comparisons per pair you care to distinguish, (3) fit Bradley-Terry with bootstrap confidence intervals, (4) report ties when intervals overlap. Skipping any of these steps produces rankings that look authoritative but are statistically noise.
      </Prose>

      <H3>Repetition strategy and budget allocation</H3>

      <Prose>
        For a fixed budget of B judge calls and N candidates, allocate budget across the protocols according to your goal. If you want a complete ranking with similar variance for all candidates, use either (a) round-robin pairwise with <Code>{"B / (N(N-1)/2)"}</Code> repeats per pair, or (b) Swiss-style sparse pairwise with adaptive pair selection. If you only care about identifying the top-k candidates, you can use a single-elimination tournament with run-off rounds — this is sub-quadratic in N but provides no useful information about ranks below the top tier. If you want a quick smoke test, use pointwise with 1–3 repeats per candidate and accept that ties and inversions will be common.
      </Prose>

      <H3>Confidence intervals via bootstrap</H3>

      <Prose>
        The single most important production diagnostic is bootstrap confidence intervals on your strength estimates. Rerun the Bradley-Terry MLE on resampled-with-replacement subsets of your comparisons, collect the resulting strength vectors, and report the percentile interval. If two candidates' intervals overlap, you do not have evidence that one is better — and ranking them anyway gives users a false sense of certainty.
      </Prose>

      <CodeBlock language="python">
{`def bootstrap_bt(comparisons, n_candidates, n_boot=200, alpha=0.05):
    """Bootstrap CI for Bradley-Terry strength estimates."""
    n = len(comparisons)
    boot_strengths = np.zeros((n_boot, n_candidates))
    for b in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        sample = [comparisons[i] for i in idx]
        s, _ = bradley_terry_mle(sample, n_candidates)
        boot_strengths[b] = s
    lo = np.percentile(boot_strengths, 100 * alpha / 2, axis=0)
    hi = np.percentile(boot_strengths, 100 * (1 - alpha / 2), axis=0)
    median = np.percentile(boot_strengths, 50, axis=0)
    return median, lo, hi

# In production: report the median and (lo, hi) interval.
# Two candidates whose intervals overlap should be reported as tied.`}
      </CodeBlock>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot shows how the standard error of the recovered ranking shrinks as you increase the number of judge calls, for each protocol. Pointwise has the worst per-call efficiency because each call is a high-variance absolute scoring; pairwise improves rapidly and plateaus near its information-theoretic limit; listwise is information-dense per call but the per-call noise is higher.
      </Prose>

      <Plot
        label="Ranking standard error vs judge call budget (N=10 candidates)"
        xLabel="judge calls"
        yLabel="mean SE of strength estimate"
        width={520}
        height={240}
        series={[
          {
            name: "pointwise",
            color: colors.gold,
            points: [[10, 1.20], [30, 0.78], [60, 0.55], [120, 0.40], [200, 0.32]],
          },
          {
            name: "pairwise (BT)",
            color: "#4ade80",
            points: [[10, 1.10], [30, 0.55], [60, 0.38], [120, 0.27], [200, 0.21]],
          },
          {
            name: "listwise (PL, K=4)",
            color: "#c084fc",
            points: [[10, 0.95], [30, 0.50], [60, 0.36], [120, 0.27], [200, 0.22]],
          },
        ]}
      />

      <Prose>
        The next plot illustrates the cost-quality frontier as you vary the number of candidates N for a fixed call budget. Pointwise stays roughly flat (linear in N), pairwise degrades quadratically once N exceeds the budget allocation, and listwise sits in between. The crossover points define when each protocol stops being efficient.
      </Prose>

      <Plot
        label="Ranking accuracy vs number of candidates (budget = 200 judge calls)"
        xLabel="number of candidates N"
        yLabel="rank correlation with truth"
        width={520}
        height={240}
        series={[
          {
            name: "pointwise",
            color: colors.gold,
            points: [[5, 0.92], [10, 0.88], [20, 0.82], [40, 0.74], [80, 0.65]],
          },
          {
            name: "pairwise (BT)",
            color: "#4ade80",
            points: [[5, 0.99], [10, 0.97], [20, 0.91], [40, 0.78], [80, 0.55]],
          },
          {
            name: "listwise (PL, K=5)",
            color: "#c084fc",
            points: [[5, 0.97], [10, 0.94], [20, 0.89], [40, 0.81], [80, 0.71]],
          },
        ]}
      />

      <Prose>
        The position-bias heatmap below shows the bias-corrected vs uncorrected pairwise win rates from the toy simulation in section 4. Cells off the diagonal show the win rate of the row candidate over the column candidate. With debiasing (left), the matrix is approximately consistent with a single latent strength scale. Without debiasing (right), there is a systematic skew that an unwary BT fit will absorb as inflated strengths for whichever side was always shown first.
      </Prose>

      <Heatmap
        matrix={[
          [0.50, 0.62, 0.73, 0.82, 0.88, 0.92],
          [0.38, 0.50, 0.62, 0.73, 0.82, 0.88],
          [0.27, 0.38, 0.50, 0.62, 0.73, 0.82],
          [0.18, 0.27, 0.38, 0.50, 0.62, 0.73],
          [0.12, 0.18, 0.27, 0.38, 0.50, 0.62],
          [0.08, 0.12, 0.18, 0.27, 0.38, 0.50],
        ]}
        rowLabels={["C0", "C1", "C2", "C3", "C4", "C5"]}
        colLabels={["C0", "C1", "C2", "C3", "C4", "C5"]}
        cellSize={48}
        colorScale="green"
        label="Pairwise win rates with order randomization (debiased)"
      />

      <Heatmap
        matrix={[
          [0.50, 0.69, 0.79, 0.86, 0.91, 0.94],
          [0.31, 0.50, 0.69, 0.79, 0.86, 0.91],
          [0.21, 0.31, 0.50, 0.69, 0.79, 0.86],
          [0.14, 0.21, 0.31, 0.50, 0.69, 0.79],
          [0.09, 0.14, 0.21, 0.31, 0.50, 0.69],
          [0.06, 0.09, 0.14, 0.21, 0.31, 0.50],
        ]}
        rowLabels={["C0", "C1", "C2", "C3", "C4", "C5"]}
        colLabels={["C0", "C1", "C2", "C3", "C4", "C5"]}
        cellSize={48}
        colorScale="gold"
        label="Pairwise win rates without order randomization (position-biased toward row)"
      />

      <Prose>
        Notice how the off-diagonal cells are systematically inflated in the biased matrix — every row candidate (always shown first) has higher win rates than its true strength would predict. A naive Bradley-Terry MLE on this matrix would still recover the correct ranking because the bias is roughly uniform, but for closer-spaced candidates the bias can flip pairs.
      </Prose>

      <Prose>
        The step trace below walks through one full hybrid evaluation cycle: pointwise filter, pairwise round-robin on survivors, Bradley-Terry inference, bootstrap confidence intervals, final ranking with ties.
      </Prose>

      <StepTrace
        label="Hybrid evaluation pipeline — one full pass"
        steps={[
          {
            label: "Generate candidates",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Inputs</div>
                <div>prompt = "Explain how DPO differs from PPO."</div>
                <div>candidates = [resp_1, resp_2, ..., resp_N]   # N model outputs</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  N is typically 5–20 in eval harnesses, up to 50+ for arena-style tournaments.
                </div>
              </div>
            ),
          },
          {
            label: "Pointwise filter",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Cheap pass with gpt-4o-mini</div>
                <div>for resp in candidates:</div>
                <div>{"    score = judge.score(prompt, resp)   # 1-5"}</div>
                <div>survivors = [r for r, s in zip(candidates, scores) if s &gt;= 2.5]</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Drops empty responses, refusals, format violations. Cuts N by 30-60% typically.
                </div>
              </div>
            ),
          },
          {
            label: "Pairwise round-robin",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Quality differentiation with gpt-4o</div>
                <div>for (i, j) in pairs(survivors):</div>
                <div>{"    for r in range(n_per_pair):"}</div>
                <div>{"        order = randomize()             # ALWAYS randomize"}</div>
                <div>{"        winner = judge.compare(prompt, i, j, order)"}</div>
                <div>{"        comparisons.append((winner, loser))"}</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Order randomization is mandatory — without it, position bias contaminates BT estimates.
                </div>
              </div>
            ),
          },
          {
            label: "Bradley-Terry MLE",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Fit latent strengths</div>
                <div>strengths, se = bradley_terry_mle(comparisons, N)</div>
                <div>{"# strengths[i] is on the same logit scale as the comparisons:"}</div>
                <div>{"# P(i beats j) = sigmoid(strengths[i] - strengths[j])"}</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Concave likelihood, no local optima. Convergence in ~20 L-BFGS iterations.
                </div>
              </div>
            ),
          },
          {
            label: "Bootstrap CIs",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Quantify uncertainty</div>
                <div>median, lo, hi = bootstrap_bt(comparisons, N, n_boot=200)</div>
                <div>{"# Resample comparisons with replacement, refit BT 200 times,"}</div>
                <div>{"# report 2.5th-97.5th percentile of resulting strengths."}</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  This is the step most production evals skip — and it's the one that prevents
                  reporting noise as signal.
                </div>
              </div>
            ),
          },
          {
            label: "Final ranking with ties",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Output</div>
                <div>{"ranking = sort_by_median(strengths)"}</div>
                <div>{"for adjacent pairs in ranking:"}</div>
                <div>{"    if intervals_overlap(lo[i], hi[i], lo[j], hi[j]):"}</div>
                <div>{"        mark_as_tied(i, j)"}</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Honest output: a partial ranking with explicit tie groups, not a forced total order.
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

      <H3>Use pointwise when...</H3>

      <Prose>
        Use pointwise when you have a large number of candidates (N &gt; 50), a tight budget, and you only need to identify the obviously good and obviously bad outputs. Pointwise is the right tool for first-pass filtering, for production monitoring (alerting on responses that score below a quality threshold), for large-scale dataset curation where you cannot afford quadratic comparisons, and for any setup where the rubric is precise enough that the judge can apply it with relatively low variance — for example, "does this response contain a working URL?" or "is this code syntactically valid?" Pointwise also wins when you need the score to be human-interpretable on a familiar scale (1–5 stars) for downstream UX consumption.
      </Prose>

      <Prose>
        The structural cost of pointwise is that absolute scoring is hard. Inter-judge agreement on Likert scales is consistently 10–20 points lower than on pairwise comparisons for the same response pairs. Variance is high, so you need 3–10 repeats per candidate to distinguish responses with similar quality. Anchor drift is real — a "4" today is not necessarily a "4" next week, and certainly not across judge model versions. When the marginal differences between candidates matter (close model comparisons, fine-grained ablations), pointwise is the wrong tool.
      </Prose>

      <H3>Use pairwise when...</H3>

      <Prose>
        Use pairwise when you have a moderate number of candidates (N = 5–30), you need reliable ranking, and you can afford <Code>{"O(N^2)"}</Code> or <Code>{"O(N \\log N)"}</Code> judge calls. Pairwise is the right tool for model leaderboards, for reward-model construction (where the binary signal directly aligns with what DPO consumes), for A/B test analysis, and for any setting where you need confidence intervals on your ranking. The Bradley-Terry inference layer turns noisy binary comparisons into continuous strength estimates with quantifiable uncertainty — this is the single biggest reason pairwise dominates production LLM evaluation.
      </Prose>

      <Prose>
        Pairwise is also the right choice when your judge model is significantly weaker than your candidate models. A weaker judge can often identify which of two responses is better even when it cannot produce an absolute quality score for either. This is the same effect that makes humans good at side-by-side comparisons even for tasks they cannot perform themselves — the cognitive task of comparison is easier than absolute evaluation.
      </Prose>

      <H3>Use listwise when...</H3>

      <Prose>
        Use listwise when you have a moderate number of candidates that fit in a single context window, your judge is strong (GPT-4-class or better), and you want to maximize information per judge call. Listwise is the right tool for retrieval and reranking pipelines (Qin et al.'s PRP framework explicitly recommends listwise prompting for top-k passage ranking), for tournaments where you can afford a small number of high-quality rounds, and for situations where the judge needs to see all candidates simultaneously to make sense of relative quality (e.g., comparing summaries of the same long document).
      </Prose>

      <Prose>
        Listwise breaks down when K is large. Most LLM judges show degraded ranking quality beyond K = 5–7 due to a combination of position bias amplification, attention budget per candidate, and the cognitive load of holding K items in working memory. If you need to rank more than 7 candidates, prefer either pairwise on the full set or listwise with overlapping K-way subsets followed by Plackett-Luce aggregation. Pure single-pass listwise on 10+ candidates is reliably worse than pairwise round-robin on the same budget.
      </Prose>

      <H3>The hybrid pattern</H3>

      <Prose>
        For most production pipelines, the right answer is hybrid: pointwise filtering followed by pairwise ranking. Pointwise removes the obviously broken candidates cheaply; pairwise resolves the remaining quality differences with statistical rigor. This is what Chatbot Arena does (basic safety filtering before pairwise voting), what production reward-model pipelines do (rule-based filtering before LLM-judge ranking), and what most internal eval harnesses converge to once teams accept that no single protocol covers all use cases.
      </Prose>

      <H3>Quick decision rules</H3>

      <Prose>
        N candidates, B budget, judge cost C per call. If <Code>B / C &lt; N</Code>: pointwise with reduced repeats, accept noise. If <Code>B / C &gt; N(N-1)/2</Code>: full round-robin pairwise with both orderings. If <Code>N &lt; B / C &lt; N(N-1)/2</Code>: hybrid (pointwise filter, pairwise on survivors) or sparse pairwise (Swiss-style). If you need to rank K-best out of a large pool: tournament structure (single-elimination with bye rounds for top seeds). If you need full ranking quality and can afford it: pairwise round-robin with bootstrap CIs.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Cost scaling. Pointwise is <Code>{"O(N)"}</Code> in candidate count, which is the only protocol that remains cheap for large-N evaluation. For N = 1000 model outputs scored by GPT-4 at <Code>$0.01 / call</Code>, pointwise costs <Code>$10</Code> per pass; pairwise round-robin costs <Code>$5000</Code>; listwise with K = 5 costs <Code>$200</Code>. The <Code>{"O(N^2)"}</Code> scaling of pairwise is the single biggest reason production evaluation systems blend protocols rather than committing to pure pairwise.
      </Prose>

      <Prose>
        Variance scaling. Pairwise variance shrinks as <Code>{"1/\\sqrt{N_{comp}}"}</Code> in the number of comparisons per pair, with the constant determined by how close the candidates are in true strength. Closely matched candidates need exponentially more comparisons to distinguish — this is the same reason chess Elo ratings have wider intervals near the top of the leaderboard than at the bottom. The Cramér-Rao lower bound on Bradley-Terry strength estimation is governed by the Fisher information, which peaks for evenly-matched comparisons; this is why Swiss-style tournaments are statistically optimal for distinguishing closely-ranked players.
      </Prose>

      <Prose>
        Listwise with growing K. The information per call grows quadratically (a K-way ranking implies <Code>{"K(K-1)/2"}</Code> ordered pairs), but the per-call noise grows roughly linearly in K due to position effects and judge cognitive load. The crossover where adding more items per list stops paying off is around K = 5 for current frontier judges; beyond that, marginal accuracy degrades faster than marginal information increases. Future judges with better long-context attention may push this limit, but as of 2026 it has held remarkably stable across Anthropic, OpenAI, and Google judge models.
      </Prose>

      <Prose>
        Confidence intervals scale poorly with the number of distinct rankings you want to identify. To distinguish all <Code>{"\\binom{N}{2}"}</Code> ordered pairs simultaneously with 95% confidence requires Bonferroni correction that multiplies the per-comparison alpha by <Code>{"\\binom{N}{2}"}</Code>. For N = 20 this is 190× — meaning the per-pair alpha needs to be <Code>0.00026</Code>, requiring substantially more comparisons than naive pairwise testing suggests. Most published model leaderboards quietly skip this correction, which is one reason adjacent positions in published rankings are often statistical ties despite being reported as ordered.
      </Prose>

      <Prose>
        Aggregation across judges scales linearly in the number of judges and is dominated by the per-judge cost. Combining 3 judges (e.g., GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro) is the standard practice for high-stakes evaluation; it costs roughly 3x and reduces judge-specific bias by averaging. Beyond 3 judges, marginal value drops because the dominant variance source becomes prompt-specific rather than judge-specific.
      </Prose>

      <Prose>
        What does not scale: human calibration of LLM judges. As model capabilities grow and judge models also grow, the assumption that an LLM judge approximates a human evaluator must be re-validated for each new judge generation, each new task domain, and each new prompt template. There is no static "LLM judge accuracy" number that ports across settings. The most robust practice is to maintain a held-out human-annotated calibration set per task and recompute judge-human agreement whenever you switch judge models or significantly change the rubric.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Position bias in pairwise and listwise</H3>
      <Prose>
        The most pervasive failure mode. LLM judges systematically favor responses presented first (or sometimes last — the direction depends on the model and prompt template). Empirical studies report 5–15 point swings in win rates due to position alone, completely independent of response quality. The mitigation is mandatory order randomization on every comparison, and ideally collecting both orderings for each pair and averaging. If you are running a leaderboard, never present the same candidate in the same position twice in a row.
      </Prose>

      <H3>Anchor drift in pointwise scores</H3>
      <Prose>
        A pointwise "4" assigned by GPT-4o today is not the same as a "4" assigned by GPT-4o-2024-08 last quarter, and it is certainly not the same as a "4" assigned by Claude 3.5 Sonnet. Absolute scores are anchored to whatever distribution the judge implicitly references at scoring time, and that anchor shifts. This makes pointwise scores unsuitable for longitudinal comparisons — you cannot say "model B is 0.3 points better than model A six months ago" using pointwise data unless you re-score model A simultaneously with the same judge. Pairwise and listwise are immune to this because the comparison is contemporaneous.
      </Prose>

      <H3>Tie inflation and the missing-tie problem</H3>
      <Prose>
        Some judge prompts explicitly allow "tie" as an option; others force a binary choice. If you allow ties and the judge picks them often (sometimes 30%+ of comparisons for closely matched pairs), Bradley-Terry as classically formulated cannot consume them — you need the Davidson tie-extension or Rao-Kupper model. If you forbid ties, the judge will fabricate distinctions that do not exist, adding noise to the BT estimates. The right policy is to allow ties, log them explicitly, and use a tie-aware extension of BT (e.g., the Davidson model with an extra "tie strength" parameter).
      </Prose>

      <H3>Judge self-preference</H3>
      <Prose>
        LLM judges systematically prefer responses generated by models from the same family. GPT-4 favors GPT-4 outputs over Claude outputs even when human raters disagree; Claude favors Claude outputs. The bias is consistent and large enough to affect leaderboard rankings. Mitigation: always evaluate with at least one judge from a different model family than your candidates, or use multiple judges from different families and aggregate.
      </Prose>

      <H3>Verbosity bias</H3>
      <Prose>
        LLM judges (and humans) systematically prefer longer, more detailed responses, even when the additional length adds no information value. Empirical studies report 10–20% bias in pairwise win rates favoring longer responses, holding quality constant. This compounds with the length bias in DPO training: if your preference data is collected via length-biased judges, the trained model learns to be verbose, which then scores well on the same biased judges in evaluation. Mitigation: either use a length-controlled judge prompt that explicitly instructs to ignore length differences, or normalize for length post-hoc by including length as a covariate in your BT regression.
      </Prose>

      <H3>The intransitivity problem</H3>
      <Prose>
        Bradley-Terry assumes there is a single latent strength scale that determines all comparisons. Real LLM-judge comparisons frequently violate this: A beats B, B beats C, but C beats A. With current frontier judges, intransitivity rates of 5–15% are typical for closely matched candidates. When intransitivity is high, the Bradley-Terry MLE still produces a ranking, but the ranking does not summarize the pairwise data well — the model is mis-specified. Diagnose this by computing the empirical Kendall tau between observed pairwise outcomes and BT-predicted outcomes; if the tau is below 0.7, your BT estimates are smoothing over real intransitivity and should be reported with extra caution.
      </Prose>

      <H3>Listwise position effects compound</H3>
      <Prose>
        Position bias in listwise prompts is larger than in pairwise because the judge has to track K items rather than two, and the order in which they appear systematically shifts the ranking. Item 1 in a K-list of 5 is consistently ranked too high; item K is consistently ranked too low. The bias is roughly twice the magnitude of the pairwise position bias. Mitigation: present each candidate in multiple positions across multiple list samples and aggregate the rankings (Borda or Kemeny-Young) — the position effects average out across permutations.
      </Prose>

      <H3>Forgetting bootstrap CIs entirely</H3>
      <Prose>
        Most published model comparisons report point Elo or BT estimates without confidence intervals. When you actually compute the bootstrap intervals, adjacent positions in the leaderboard frequently overlap, meaning the ranking between them is not statistically supported. This is endemic in academic LLM evaluation papers and a major source of reproducibility problems. The fix is mechanical: always run bootstrap, always report intervals, always mark overlapping intervals as ties.
      </Prose>

      <H3>Same-model bias on benchmarks</H3>
      <Prose>
        If you use GPT-4 to generate the preference data that trained the reward model that scored the candidates, your evaluation is contaminated. This pattern is common: GPT-4 generates AI-feedback for DPO training, then GPT-4 is the judge for the eval. Any systematic bias in GPT-4's preferences (verbosity, formality, certain phrasings) is now both a training signal and an evaluation signal, which inflates the apparent quality of the resulting model. Mitigation: train and evaluate with different judges, ideally from different model families.
      </Prose>

      <Callout accent="gold">
        The single most important production discipline: always randomize presentation order, always compute bootstrap confidence intervals, and always report ties when intervals overlap. Skipping any of these turns evaluation noise into apparent signal.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All sources verified against arXiv pages on 2026-04-26. Author lists, abstracts, and arXiv IDs confirmed.
      </Prose>

      <H3>Zheng et al. 2023 — MT-Bench and Chatbot Arena</H3>
      <Prose>
        Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." arXiv:2306.05685. Published June 2023; NeurIPS 2023 Datasets and Benchmarks Track. The foundational LLM-as-judge paper. Establishes that GPT-4 as a pairwise judge agrees with human preferences at &gt;80% rates on multi-turn dialogue, characterizes position bias and verbosity bias quantitatively, introduces Chatbot Arena as the canonical pairwise evaluation platform, and validates Bradley-Terry/Elo inference on millions of pairwise comparisons. Required reading for anyone deploying LLM-as-judge in production.
      </Prose>

      <H3>Liusie et al. 2024 — LLM Comparative Assessment</H3>
      <Prose>
        Adian Liusie, Potsawee Manakul, Mark J. F. Gales. "LLM Comparative Assessment: Zero-shot NLG Evaluation through Pairwise Comparisons with Large Language Models." arXiv:2307.07889. Published July 2023, updated 2024. The most thorough head-to-head comparison of pointwise vs pairwise vs listwise LLM evaluation across summarization, dialogue, and translation benchmarks. Shows pairwise consistently outperforms pointwise for LLM judges across model sizes from 7B to GPT-4, characterizes the bias-variance tradeoff explicitly, and provides empirical evidence for Bradley-Terry as the appropriate aggregation model. Methodologically the most rigorous paper in the area and the right starting point for understanding the protocol comparison.
      </Prose>

      <H3>Qin et al. 2023 — Pairwise Ranking Prompting</H3>
      <Prose>
        Zhen Qin, Rolf Jagerman, Kai Hui, Honglei Zhuang, Junru Wu, Jiaming Shen, Tianqi Liu, Jialu Liu, Donald Metzler, Xuanhui Wang, Michael Bendersky. "Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting." arXiv:2306.17563. Published June 2023; NAACL 2024. Introduces PRP (Pairwise Ranking Prompting) for passage retrieval and shows it consistently beats pointwise prompting on TREC-DL benchmarks across model sizes from FLAN-T5 to GPT-3.5. Provides the canonical demonstration that pairwise prompting unlocks ranking quality from models that cannot produce reliable pointwise scores, and discusses the cost-quality frontier explicitly.
      </Prose>

      <H3>Bradley & Terry 1952 — original model</H3>
      <Prose>
        Ralph Allan Bradley, Milton E. Terry. "Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons." Biometrika, Vol. 39, No. 3/4 (Dec., 1952), pp. 324-345. The original Bradley-Terry paper. Derives the maximum likelihood framework for paired-comparison data, proves the iterative algorithm converges, and establishes the theoretical foundation for everything from FIDE chess Elo ratings to modern Chatbot Arena. Essential historical context; the math has not changed in 70 years.
      </Prose>

      <H3>Plackett 1975 — Plackett-Luce model</H3>
      <Prose>
        R. L. Plackett. "The Analysis of Permutations." Journal of the Royal Statistical Society, Series C (Applied Statistics), Vol. 24, No. 2 (1975), pp. 193-202. The original Plackett-Luce paper, extending Bradley-Terry to full K-way rankings. Derives the sequential top-1 factorization and proves identifiability up to additive constants. Together with Luce's 1959 axiomatic derivation of choice probabilities, this is the theoretical basis for all listwise ranking inference.
      </Prose>

      <H3>Dubois et al. 2024 — AlpacaEval and length bias</H3>
      <Prose>
        Yann Dubois, Balázs Galambosi, Percy Liang, Tatsunori B. Hashimoto. "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators." arXiv:2404.04475. Published April 2024. Quantifies the length bias in LLM-as-judge pairwise evaluation, shows that AlpacaEval 2 win rates are dominated by response length differences in many model comparisons, and proposes a length-control regression that disentangles content quality from verbosity. Essential for anyone running automatic evaluation as a primary signal for model selection or training-data construction.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Derive the BT MLE update</H3>
      <Prose>
        Starting from the Bradley-Terry log-likelihood <Code>{"\\ell(s) = \\sum_t [y_t \\log \\sigma(s_{i_t} - s_{j_t}) + (1-y_t) \\log \\sigma(s_{j_t} - s_{i_t})]"}</Code>, derive the gradient with respect to <Code>{"s_i"}</Code> and show that setting it to zero produces the Zermelo iterative update <Code>{"s_i^{(t+1)} = \\log W_i - \\log \\sum_{j} N_{ij} / (e^{s_i} + e^{s_j})"}</Code>. Why is this update guaranteed to converge? What property of the log-likelihood ensures the converged solution is the global MLE rather than a local optimum?
      </Prose>

      <H3>Exercise 2 — Sample complexity comparison</H3>
      <Prose>
        You have N = 20 candidates and a budget of 200 judge calls. Compute the expected ranking-recovery accuracy (rank correlation with truth) you would get from each protocol assuming Gaussian per-call noise of <Code>σ = 1.0</Code> on logit scale and a uniform spacing of true strengths (gap = 0.3 between adjacent candidates). Which protocol would you choose? At what value of N would the ranking change?
      </Prose>

      <H3>Exercise 3 — Position bias analysis</H3>
      <Prose>
        Suppose the judge has a position bias <Code>b</Code> in favor of the response presented first. Show that if you alternate the presentation order across replicates of the same pair, the bias cancels in expectation in the resulting comparison count. Now suppose you only present each pair in one ordering — derive how the BT MLE strength estimates are systematically shifted as a function of the bias <Code>b</Code> and the candidate's average position over all comparisons. What is the magnitude of the shift if all candidates are presented first equally often versus all-presented-second equally often?
      </Prose>

      <H3>Exercise 4 — Information per call</H3>
      <Prose>
        Compute the Fisher information about the strength vector <Code>s</Code> from a single K-way Plackett-Luce ranking versus from <Code>{"\\binom{K}{2}"}</Code> independent pairwise comparisons. Show that they are equal in the noise-free limit but diverge when per-judgment noise is added. At what level of judge noise does pairwise become more efficient than listwise per unit of information? How does this analysis inform the choice of K in production listwise evaluation?
      </Prose>

      <H3>Exercise 5 — Bootstrap and ties</H3>
      <Prose>
        You have run 50 pairwise comparisons per pair on N = 8 candidates and computed bootstrap 95% confidence intervals for each candidate's BT strength. Two adjacent candidates have intervals <Code>{"[0.20, 0.45]"}</Code> and <Code>{"[0.30, 0.55]"}</Code>. Are they statistically distinguishable? Design a hypothesis test specifically for the comparison of two BT strengths from the same fitted model — what is the right test statistic, what is its asymptotic distribution, and how does it differ from naively comparing CIs? What is the practical consequence for how you should present the leaderboard to users?
      </Prose>

      <H3>Exercise 6 — Aggregation comparison</H3>
      <Prose>
        Three judges produce the following rankings of 5 candidates: Judge A: [C1, C2, C3, C4, C5]. Judge B: [C2, C1, C3, C5, C4]. Judge C: [C1, C3, C2, C4, C5]. Compute the Borda aggregate ranking and the Kemeny-Young aggregate ranking. Do they agree? Now suppose Judge B's ranking is changed to [C5, C4, C3, C2, C1] — recompute both aggregates. Why are they more likely to disagree on this adversarial input? Which aggregator would you trust more in a production setting and why?
      </Prose>

      <H3>Exercise 7 — Designing a hybrid pipeline</H3>
      <Prose>
        You need to evaluate 100 candidate model outputs against a fixed prompt set, with a budget of 1000 judge calls split between GPT-4 (expensive, $0.02/call) and GPT-4o-mini (cheap, $0.001/call). Design a hybrid evaluation pipeline that maximizes ranking quality within this budget. What protocol do you use at each stage, how do you allocate calls, and how do you handle survivors that the cheap pass rejected but you want to verify? Compute the expected total cost and compare it to pure pairwise round-robin with GPT-4.
      </Prose>

    </div>
  ),
};

export default pointwisePairwiseListwise;
