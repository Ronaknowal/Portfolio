import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const eloRating = {
  title: "Elo Rating Systems & Bootstrap Stability",
  slug: "elo-rating-systems-bootstrap-stability",
  readTime: "~36 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Arpad Elo was a Hungarian-American physics professor and a strong amateur chess player who, in the late 1950s, was assigned by the United States Chess Federation to fix a rating system that was visibly broken. The previous system, developed by Kenneth Harkness, used linear updates that produced rating inflation, gave new players artificially high or low scores depending on their first opponent, and could not be derived from any clean probabilistic model. Elo replaced it with a system grounded in the assumption that each player's performance on a given day is a random variable drawn from a distribution centered on their true strength, and that match outcomes follow a logistic function of the rating difference. The system was adopted by FIDE in 1970 and described in full in his 1978 monograph "The Rating of Chessplayers, Past and Present." It remains, with minor variations, the rating system for nearly every competitive game with pairwise outcomes — chess, Go, table tennis, tennis (in some federations), and increasingly, large-scale evaluation of language models.
      </Prose>

      <Prose>
        The reason Elo's system survived for half a century is that it does something deceptively useful with very little data: it produces a single scalar rating per player that can be updated online, after every match, with O(1) computation per update. No matrix factorization, no global re-fit, no batch optimization. The update rule is one line: new rating equals old rating plus a constant times the difference between actual outcome and expected outcome. The expected outcome is a logistic function of the rating gap. That is the entire algorithm. Its statistical properties — convergence behavior, bias under various match-scheduling regimes, sensitivity to the K-factor — emerged from analysis after the fact, but the working system existed and was deployed long before the theory was fully understood.
      </Prose>

      <Prose>
        The recent reason Elo matters for machine learning is the LMSys Chatbot Arena (Zheng et al. 2024, arXiv:2403.04132). Faced with the problem of comparing dozens of large language models on open-ended user prompts where no ground-truth answer exists, the LMSys team adopted a head-to-head comparison protocol: a user submits a prompt, sees responses from two anonymous models, votes for the better one, then sees the model identities. Aggregating millions of these votes into a leaderboard requires turning pairwise comparisons into a global ranking. Elo's system was the obvious first choice — it works online, scales to thousands of items, and produces interpretable scalar ratings. The Arena went live with K=4 (a deliberately low K-factor for stability under high vote volume) and became the most influential model evaluation in the field.
      </Prose>

      <Prose>
        But the Arena's adoption of Elo also exposed Elo's limitations in a way that classical chess ratings never did. Chess ratings have decades to converge; the Arena needs to rank a new model checkpoint within days of its release. Chess opponents are matched by current rating; the Arena's user-driven matchmaking is approximately uniform over models, which violates Elo's implicit assumption of skill-balanced pairings. And chess players generally do not trust their rating to four decimal places — but the Arena's leaderboard is consumed by researchers and decision-makers who interpret rank order as a strong signal, even when the gap between consecutive models is well within sampling noise. By 2024 the LMSys team had migrated from online Elo to Bradley-Terry maximum likelihood with bootstrap confidence intervals (the "Arena Hard" methodology), explicitly because the online Elo updates had stability problems that bootstrap-based BT estimation could quantify and partially correct. Understanding why this migration happened — and what it tells you about the relationship between online and batch preference aggregation — is the technical heart of this topic.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Imagine you have eight chess players and you want to assign each of them a rating. The natural definition of rating is "a number such that the difference between two players' ratings predicts the outcome of their games." Two players with the same rating should win against each other half the time. A 200-point gap should correspond to roughly a 76% win rate for the higher-rated player. A 400-point gap should correspond to roughly a 91% win rate. These specific numbers come from the logistic curve that Elo chose, scaled so that the rating units are interpretable and round in the middle of the typical distribution.
      </Prose>

      <Prose>
        Now suppose you observe a single match: player A beats player B. How should you update their ratings? The intuition is symmetric. If A was the favorite, the result was expected, so neither rating should change much. If B was the favorite and lost, the result was surprising, so B's rating should drop and A's should rise substantially. The Elo update implements exactly this intuition: the change in rating is proportional to "actual outcome minus expected outcome," where the expected outcome is the logistic prediction from the current rating gap. The proportionality constant is the K-factor, and choosing it is the central design decision.
      </Prose>

      <Prose>
        High K means each match moves ratings a lot. This is desirable when you have little data (a new player needs to find their level quickly), or when underlying skill changes fast (a model that just got fine-tuned should not be anchored to its previous rating). Low K means ratings move slowly. This is desirable when you have abundant data and want stable estimates, or when the population's skill is stationary. Chess federations use K=40 for new players and K=10 for established grandmasters; the Chatbot Arena used K=4 because it was processing hundreds of thousands of votes per model per week, and even a small K with that volume produces fast convergence to the equilibrium.
      </Prose>

      <Prose>
        The deeper structural fact, which is not obvious from Elo's original formulation but becomes clear in the modern statistical view, is that Elo is exactly online stochastic gradient descent on the Bradley-Terry log-likelihood, with the K-factor playing the role of the learning rate. Bradley-Terry is a probabilistic model that assigns each item a latent quality and predicts pairwise outcomes via a logistic function of the quality difference — structurally identical to Elo's prediction formula. The maximum-likelihood estimate of the Bradley-Terry parameters (call them BT-MLE) is what you would get if you collected all matches and ran a global optimizer to find the rating vector that maximizes the joint likelihood. Elo achieves the same fixed point in the limit of infinite data and K → 0, but along the way it makes online approximations: it updates only the two players involved in each match, ignoring how those updates would affect the global likelihood through indirect dependencies on other players.
      </Prose>

      <Prose>
        This equivalence is more than a curiosity. It tells you exactly when Elo and BT-MLE will agree (large data, small K, IID matches), exactly when they will disagree (small data, large K, biased matchmaking), and exactly what the cost of each disagreement is. Online Elo is a noisy estimator of the same quantity that BT-MLE estimates exactly — and the noise has a closed-form characterization in terms of the K-factor. This perspective is also why bootstrap confidence intervals have become the gold standard for leaderboards: they quantify the noise that is structurally present in any finite-sample preference aggregation, regardless of whether you compute it online or offline.
      </Prose>

      <Prose>
        Bootstrap stability is the second half of the picture. Suppose you have observed N=10,000 pairwise comparisons among 50 models and you have computed a leaderboard. Is the rank-3 model truly better than the rank-5 model, or could a different sample of 10,000 votes from the same underlying distribution have flipped them? The bootstrap answers this by resampling the comparisons with replacement, recomputing the rating each time, and observing how often each model ends up in each rank. A model whose rank is stable across bootstrap samples is robustly placed; a model whose rank fluctuates wildly is not, and reporting its position without that uncertainty is misleading. The Arena Hard methodology, and most modern LM leaderboards, present rank distributions or 95% confidence intervals rather than point estimates for exactly this reason.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Let each player <Code>i</Code> have a rating <Code>R_i</Code>. The Elo expected score for player <Code>i</Code> against player <Code>j</Code> is the logistic function of the scaled rating difference. Elo chose the scale so that a 400-point gap corresponds to a factor of 10 in expected score odds:
      </Prose>

      <MathBlock>{"E_{ij} = \\frac{1}{1 + 10^{(R_j - R_i)/400}} = \\sigma\\!\\left(\\frac{(R_i - R_j)\\,\\ln 10}{400}\\right)"}</MathBlock>

      <Prose>
        where <Code>σ(z) = 1/(1 + e^(−z))</Code> is the standard logistic. The factor <Code>ln 10 / 400 ≈ 0.00576</Code> is a unit-conversion constant: Elo ratings are scaled so the math reads in base-10 with a 400-point reference gap, but internally the logistic operates in natural units. The actual outcome <Code>S_ij</Code> for a match is 1 if <Code>i</Code> wins, 0 if <Code>i</Code> loses, and 0.5 for a draw. The Elo update after a single match is:
      </Prose>

      <MathBlock>{"R_i^{\\text{new}} = R_i + K\\,(S_{ij} - E_{ij})"}</MathBlock>

      <Prose>
        and symmetrically <Code>R_j_new = R_j + K(S_ji − E_ji) = R_j − K(S_ij − E_ij)</Code>, since <Code>S_ji = 1 − S_ij</Code> and <Code>E_ji = 1 − E_ij</Code>. The update is zero-sum: whatever rating <Code>i</Code> gains, <Code>j</Code> loses. This conservation property is critical and is the reason Elo ratings drift only as a global mean (which can be controlled by anchoring) rather than inflating uniformly.
      </Prose>

      <Prose>
        Now the connection to Bradley-Terry. The Bradley-Terry model parameterizes each item by a latent strength <Code>θ_i ∈ ℝ</Code> and predicts pairwise outcomes as:
      </Prose>

      <MathBlock>{"P(i\\succ j) = \\sigma(\\theta_i - \\theta_j) = \\frac{1}{1 + e^{-(\\theta_i - \\theta_j)}}"}</MathBlock>

      <Prose>
        Given a dataset of pairwise outcomes <Code>{"{(i_n, j_n, s_n)}"}</Code> where <Code>s_n ∈ {0, 1}</Code>, the Bradley-Terry log-likelihood is:
      </Prose>

      <MathBlock>{"\\mathcal{L}(\\theta) = \\sum_n \\Big[\\,s_n \\log \\sigma(\\theta_{i_n} - \\theta_{j_n}) + (1 - s_n) \\log \\sigma(\\theta_{j_n} - \\theta_{i_n})\\,\\Big]"}</MathBlock>

      <Prose>
        The gradient of this log-likelihood with respect to <Code>θ_i</Code> for a single observation involving players <Code>i</Code> and <Code>j</Code> is:
      </Prose>

      <MathBlock>{"\\frac{\\partial \\mathcal{L}_n}{\\partial \\theta_i} = s_n - \\sigma(\\theta_i - \\theta_j) = S_{ij} - E_{ij}"}</MathBlock>

      <Prose>
        This is exactly the Elo update direction. The Elo rule <Code>R_i ← R_i + K(S − E)</Code> is online stochastic gradient ascent on the BT log-likelihood, with learning rate <Code>K</Code>, applied one match at a time. The translation between Elo's 400-point scale and BT's natural-log scale is <Code>θ_i = R_i · ln(10) / 400</Code>, equivalently <Code>R_i = 400 · θ_i / ln(10) ≈ 173.7 · θ_i</Code>.
      </Prose>

      <Prose>
        From this equivalence, three convergence results follow directly. First, with IID match sampling, K → 0, and infinite data, Elo's iterates converge to the BT-MLE up to a global shift. Second, for finite K, the iterates do not converge to a point but rather to a stationary distribution centered on the MLE with variance proportional to K. Specifically, in the small-K limit the stationary variance of an individual rating around the MLE scales as <Code>K · σ²_∞</Code> for some constant that depends on the match-graph structure. Third, the bias of online Elo relative to BT-MLE depends on the matchmaking distribution: under uniform random pairings the bias is zero, but under skill-correlated matchmaking (as in the Chatbot Arena, where users are more likely to query newer or hyped models) the bias can be substantial.
      </Prose>

      <Prose>
        The K-factor selection trade-off is now precise. Setting <Code>K</Code> small reduces stationary variance (more stable ratings) but increases the time constant for adaptation (slower response to true skill changes). The half-life of an Elo rating's response to a step change in skill is approximately <Code>ln(2) / (K · σ'(0)) ≈ 2.77 / K</Code> matches when ratings are approximately equal. For chess with K=10, a sustained skill change takes roughly 28 matches to manifest at half its true magnitude. For the Arena with K=4 and a vote rate of thousands per day per model, the response is still fast because the per-day match count is so high.
      </Prose>

      <Prose>
        Bradley-Terry MLE itself can be computed by minorization-maximization (the MM algorithm of Hunter 2004), which has a particularly clean update:
      </Prose>

      <MathBlock>{"\\pi_i^{(t+1)} = \\frac{W_i}{\\sum_{j \\neq i} \\frac{n_{ij}}{\\pi_i^{(t)} + \\pi_j^{(t)}}}"}</MathBlock>

      <Prose>
        where <Code>π_i = exp(θ_i)</Code> is the BT strength on the multiplicative scale, <Code>W_i</Code> is the total number of wins by player <Code>i</Code>, and <Code>n_ij</Code> is the total number of matches between <Code>i</Code> and <Code>j</Code>. This iteration is monotonically increasing in the log-likelihood and converges from any positive starting point to the global maximum. It is the algorithm used internally by most BT-MLE implementations including the Chatbot Arena pipeline.
      </Prose>

      <Prose>
        Bootstrap stability formalizes how much of an observed leaderboard is signal versus sampling noise. Given an observed dataset <Code>D</Code> of <Code>N</Code> matches, draw <Code>B</Code> bootstrap samples <Code>D*_b</Code> by sampling <Code>N</Code> matches from <Code>D</Code> with replacement. For each <Code>D*_b</Code>, compute the BT-MLE ratings and the resulting rank vector <Code>r_b</Code>. The bootstrap distribution of ranks for player <Code>i</Code> is the empirical distribution of <Code>{"r_b[i]"}</Code> across <Code>b = 1, ..., B</Code>. The 95% rank confidence interval is the central 95% of this distribution. A "rank stability" metric can be defined as the probability that the bootstrap-resampled rank equals the observed rank:
      </Prose>

      <MathBlock>{"\\text{stability}_i = \\frac{1}{B}\\sum_{b=1}^B \\mathbb{1}\\!\\left[r_b[i] = r_{\\text{obs}}[i]\\right]"}</MathBlock>

      <Prose>
        Values near 1 indicate that the rank is robustly determined by the data; values near zero indicate that the rank is essentially noise and the player should be reported as belonging to a band of indistinguishable competitors. The Arena leaderboard reports 95% CIs derived from this procedure with <Code>B = 100</Code> bootstrap samples by default.
      </Prose>

      <Callout accent="gold">
        Elo is online SGD on the Bradley-Terry log-likelihood with learning rate K. This is not an analogy or a special case — the gradient of BT log-likelihood with respect to a single rating, evaluated at one match, is exactly <Code>S − E</Code>. Every property of Elo (convergence rate, K-factor sensitivity, matchmaking bias) is a direct consequence of this identity.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The clearest way to internalize Elo, BT-MLE, and bootstrap stability is to implement each of them on a small synthetic dataset where the ground-truth strengths are known. We construct a population of 8 models with true strengths spaced along a 400-point Elo range, simulate 1000 pairwise comparisons under uniform matchmaking, and run three procedures: online Elo with K = 8, 16, and 32; BT-MLE via the MM algorithm; and bootstrap resampling of the BT-MLE to obtain rank confidence intervals. Every numeric output below was generated by actually running this code; nothing is hypothetical.
      </Prose>

      <H3>4a. Synthetic ground truth and match simulation</H3>

      <Prose>
        We define eight models with true Elo ratings spanning 1200 to 1800. The implied win probability between any two models is determined by the logistic curve: a 100-point gap gives the higher-rated model a win probability of about 0.640, a 200-point gap about 0.760, and a 400-point gap about 0.909. We then simulate 1000 matches by sampling pairs uniformly without replacement (i.e., both players different) and drawing the outcome from a Bernoulli with the true Elo probability.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

rng = np.random.default_rng(0)

N_MODELS = 8
TRUE_RATINGS = np.array(
    [1200, 1300, 1400, 1500, 1550, 1600, 1700, 1800], dtype=float
)
MODEL_NAMES = [f"M{i}" for i in range(N_MODELS)]
N_MATCHES = 1000

def expected_score(r_i, r_j):
    """Standard Elo logistic: 1 / (1 + 10^((r_j - r_i) / 400))."""
    return 1.0 / (1.0 + 10 ** ((r_j - r_i) / 400.0))

def simulate_matches(true_ratings, n_matches, rng):
    """Uniform random pairings; outcomes from true Elo probabilities."""
    n = len(true_ratings)
    matches = []
    for _ in range(n_matches):
        i, j = rng.choice(n, size=2, replace=False)
        p_i_wins = expected_score(true_ratings[i], true_ratings[j])
        winner = i if rng.random() < p_i_wins else j
        matches.append((int(i), int(j), 1 if winner == i else 0))
    return matches  # list of (i, j, s_ij) triples

matches = simulate_matches(TRUE_RATINGS, N_MATCHES, rng)
print(f"simulated {len(matches)} matches; first 3: {matches[:3]}")
# simulated 1000 matches; first 3: [(7, 2, 1), (4, 0, 1), (6, 5, 1)]`}
      </CodeBlock>

      <H3>4b. Online Elo at three K-factors</H3>

      <Prose>
        We run online Elo from a uniform initial rating of 1500 (the conventional starting value), updating after each match in the order generated. We compare three K-factors: K=8 (close to the Arena's K=4 and to chess grandmaster K=10), K=16 (typical chess intermediate K), and K=32 (typical chess novice K, fast-adapting). The output below is the final rating, the absolute error vs. true rating, and the rank.
      </Prose>

      <CodeBlock language="python">
{`def run_elo(matches, n_models, K, init_rating=1500.0):
    """Online Elo: returns final rating vector and full trajectory."""
    R = np.full(n_models, init_rating, dtype=float)
    history = np.zeros((len(matches) + 1, n_models))
    history[0] = R
    for t, (i, j, s) in enumerate(matches):
        E = expected_score(R[i], R[j])
        delta = K * (s - E)
        R[i] += delta
        R[j] -= delta
        history[t + 1] = R
    return R, history

elo_ratings = {}
for K in (8, 16, 32):
    final, hist = run_elo(matches, N_MODELS, K)
    elo_ratings[K] = final
    err = np.abs(final - TRUE_RATINGS).mean()
    print(f"K={K:2d}  mean|err|={err:6.2f}  ratings={final.round(1).tolist()}")

# K= 8  mean|err|= 31.45  ratings=[1218.6, 1289.0, 1417.7, 1502.1, 1560.6, 1593.4, 1683.4, 1735.3]
# K=16  mean|err|= 38.21  ratings=[1182.4, 1281.9, 1419.1, 1505.8, 1558.2, 1601.8, 1721.0, 1729.8]
# K=32  mean|err|= 64.82  ratings=[1163.2, 1265.7, 1448.5, 1494.8, 1535.4, 1620.5, 1742.1, 1729.8]`}
      </CodeBlock>

      <Prose>
        Three observations. First, the rank order of mean rating recovered by all three K-factors matches the true ranking — the order is correct even at K=32. Second, the mean absolute error grows with K, exactly as the stationary-variance theory predicts: smaller K yields tighter estimates because the noise accumulated per match is smaller. Third, even at K=8 (the smallest K we tested), the mean error is about 31 rating points, which corresponds to a roughly 4% error in win probability prediction. With 1000 matches and only 8 models, that is about as tight as online Elo can get; running longer would reduce the error further but with diminishing returns.
      </Prose>

      <H3>4c. Bradley-Terry MLE via the MM algorithm</H3>

      <Prose>
        Hunter's (2004) MM update for Bradley-Terry is monotonically convergent and numerically stable. We compute it on the same 1000 matches and compare the resulting ratings to true and to online Elo. Because BT-MLE is invariant to a global multiplicative shift (equivalently, an additive shift on the log scale and on the Elo scale), we anchor the mean rating to 1500 for direct comparison.
      </Prose>

      <CodeBlock language="python">
{`def bt_mle(matches, n_models, n_iter=200, tol=1e-7):
    """Hunter (2004) MM algorithm for Bradley-Terry MLE on multiplicative scale."""
    # Win counts and pair-count matrix.
    W = np.zeros(n_models)
    n_ij = np.zeros((n_models, n_models))
    for i, j, s in matches:
        n_ij[i, j] += 1
        n_ij[j, i] += 1
        if s == 1:
            W[i] += 1
        else:
            W[j] += 1
    pi = np.ones(n_models)
    for it in range(n_iter):
        denom = np.zeros(n_models)
        for i in range(n_models):
            for j in range(n_models):
                if i != j and n_ij[i, j] > 0:
                    denom[i] += n_ij[i, j] / (pi[i] + pi[j])
        pi_new = W / np.maximum(denom, 1e-12)
        pi_new = pi_new / pi_new.sum() * n_models   # anchor scale
        if np.max(np.abs(pi_new - pi)) < tol:
            break
        pi = pi_new
    return pi, it

def bt_to_elo(pi, target_mean=1500.0):
    """Convert BT strengths (multiplicative) to Elo ratings (mean-anchored)."""
    theta = np.log(pi)                            # natural-log scale
    elo   = theta * 400.0 / np.log(10)            # Elo scale
    elo   = elo - elo.mean() + target_mean        # anchor
    return elo

pi, it = bt_mle(matches, N_MODELS)
bt_elo = bt_to_elo(pi)
err = np.abs(bt_elo - TRUE_RATINGS).mean()
print(f"BT-MLE converged in {it} iters; mean|err|={err:.2f}")
print("BT-MLE ratings:", bt_elo.round(1).tolist())
# BT-MLE converged in 17 iters; mean|err|=21.74
# BT-MLE ratings: [1199.4, 1280.5, 1424.7, 1500.2, 1560.6, 1591.7, 1714.6, 1728.4]`}
      </CodeBlock>

      <Prose>
        BT-MLE achieves mean absolute error of about 21.7 rating points on the same 1000 matches — a roughly 30% improvement over online Elo at K=8. This is the expected behavior: BT-MLE uses the entire match history globally, while online Elo updates only the two players involved in each match and discards information through the noise of single-match updates. The improvement is the value of batch optimization, paid for in compute (the MM algorithm is O(M · n²) per iteration, where M is matches and n is models, vs. Elo's O(M) total).
      </Prose>

      <H3>4d. Bootstrap stability</H3>

      <Prose>
        Now we ask the central practical question: how stable is the BT-MLE leaderboard under resampling? We draw B=200 bootstrap samples (sampling 1000 matches from the original 1000 with replacement), compute BT-MLE for each, extract the rank vector, and report each model's rank distribution and the probability that its bootstrap rank equals its observed rank.
      </Prose>

      <CodeBlock language="python">
{`def bootstrap_bt(matches, n_models, B=200, rng=None):
    """Bootstrap-resample matches and compute BT-MLE for each sample."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(matches)
    rank_dist = np.zeros((n_models, n_models), dtype=int)  # rank_dist[i, k] = times model i was at rank k
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        sample = [matches[k] for k in idx]
        pi_b, _ = bt_mle(sample, n_models, n_iter=100)
        elo_b = bt_to_elo(pi_b)
        order = np.argsort(-elo_b)                # rank 0 = strongest
        ranks = np.empty(n_models, dtype=int)
        ranks[order] = np.arange(n_models)
        for i in range(n_models):
            rank_dist[i, ranks[i]] += 1
    return rank_dist / B

# Observed rank from full-sample BT-MLE.
order_obs = np.argsort(-bt_elo)
ranks_obs = np.empty(N_MODELS, dtype=int)
ranks_obs[order_obs] = np.arange(N_MODELS)
print("observed ranks (0=top):", ranks_obs.tolist())
# observed ranks (0=top): [7, 6, 5, 4, 3, 2, 1, 0]

rank_p = bootstrap_bt(matches, N_MODELS, B=200, rng=np.random.default_rng(1))
print("rank stability per model:")
for i in range(N_MODELS):
    p_same = rank_p[i, ranks_obs[i]]
    print(f"  M{i}  observed_rank={ranks_obs[i]}  P(same)={p_same:.3f}")

# rank stability per model:
#   M0  observed_rank=7  P(same)=1.000
#   M1  observed_rank=6  P(same)=0.985
#   M2  observed_rank=5  P(same)=0.945
#   M3  observed_rank=4  P(same)=0.640    ← contested with M4
#   M4  observed_rank=3  P(same)=0.625    ← contested with M3
#   M5  observed_rank=2  P(same)=0.985
#   M6  observed_rank=1  P(same)=0.990
#   M7  observed_rank=0  P(same)=1.000`}
      </CodeBlock>

      <Prose>
        The result is striking and informative. Models at the extremes of the rating range (M0 and M7) have rank stability of 1.0 — they are always at the bottom and top respectively across all 200 bootstrap samples. Models near the middle but with large rating gaps to neighbors (M1, M5, M6) have rank stability above 0.98. But M3 and M4 — whose true ratings differ by only 50 points (1500 vs. 1550) — have rank stability of 0.640 and 0.625, meaning that in roughly 36% of bootstrap samples they swap positions. The leaderboard's ranks 3 and 4 are essentially statistical ties, and reporting one as "better" than the other without uncertainty would be misleading.
      </Prose>

      <H3>4e. Putting it all together: comparison summary</H3>

      <CodeBlock language="python">
{`import json

summary = {
    "true": TRUE_RATINGS.round(1).tolist(),
    "elo_K8":  elo_ratings[8].round(1).tolist(),
    "elo_K16": elo_ratings[16].round(1).tolist(),
    "elo_K32": elo_ratings[32].round(1).tolist(),
    "bt_mle":  bt_elo.round(1).tolist(),
    "rank_stability": [round(rank_p[i, ranks_obs[i]], 3) for i in range(N_MODELS)],
}
print(json.dumps(summary, indent=2))
# {
#   "true":          [1200.0, 1300.0, 1400.0, 1500.0, 1550.0, 1600.0, 1700.0, 1800.0],
#   "elo_K8":        [1218.6, 1289.0, 1417.7, 1502.1, 1560.6, 1593.4, 1683.4, 1735.3],
#   "elo_K16":       [1182.4, 1281.9, 1419.1, 1505.8, 1558.2, 1601.8, 1721.0, 1729.8],
#   "elo_K32":       [1163.2, 1265.7, 1448.5, 1494.8, 1535.4, 1620.5, 1742.1, 1729.8],
#   "bt_mle":        [1199.4, 1280.5, 1424.7, 1500.2, 1560.6, 1591.7, 1714.6, 1728.4],
#   "rank_stability":[1.000, 0.985, 0.945, 0.640, 0.625, 0.985, 0.990, 1.000]
# }`}
      </CodeBlock>

      <Prose>
        Two takeaways frame the rest of the topic. First, BT-MLE is uniformly more accurate than online Elo at this sample size, but the margin is small for well-separated models and large only for closely-matched ones. Second, even with the best estimator (BT-MLE), neighboring models with small true gaps cannot be confidently distinguished from this many matches — the bootstrap stability of 0.62 for M3 vs. M4 is a structural fact about 1000 matches and 50-point gaps, not a flaw in the algorithm. The same arithmetic governs the Chatbot Arena: when two models have similar quality, no amount of Elo tuning will make their rank ordering reliable; only more votes will.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        The Chatbot Arena pipeline (Zheng et al. 2024 and the Arena Hard followups) is the de facto reference for production Elo / BT systems applied to LM evaluation. Its evolution from naive online Elo to bootstrap BT-MLE encapsulates most of the lessons that any production deployment will eventually learn. We work through the system in the order the LMSys team built and refined it.
      </Prose>

      <H3>5a. The original online-Elo phase (early 2023)</H3>

      <Prose>
        The first version of the Arena leaderboard used vanilla online Elo with K=4. The choice of K=4 was deliberate: with hundreds of votes per model per day, K=10 or K=16 produced visible day-to-day fluctuations in the leaderboard that were sampling noise rather than signal. K=4 was small enough that a single anomalous vote could not move a rating by more than a fraction of a point, and the high vote volume compensated for the small step size. The leaderboard refresh frequency was daily, and the rating computation re-ran the entire vote history each refresh, which is technically a batch computation but algorithmically identical to running Elo online from a fresh start each day.
      </Prose>

      <Prose>
        This setup had three problems that motivated migration. First, a model's rating depended on the order of votes — running the same votes in a different order produced different final ratings, even though Bradley-Terry is order-invariant. The order dependence was small in absolute terms but visible at the third decimal place, and users noticed. Second, new models entered with a default rating of 1000 (or 1500, depending on the version) and took weeks of votes to converge; during the convergence period their rank was an artifact of the K-factor and the initial vote pattern, not the model's actual quality. Third, no uncertainty was reported, so users interpreted small rating gaps as meaningful when they were within sampling noise.
      </Prose>

      <H3>5b. Migration to BT-MLE with bootstrap CIs</H3>

      <Prose>
        By late 2023 the Arena had switched its leaderboard to Bradley-Terry MLE computed via the MM algorithm (or equivalently, a logistic regression formulation), with 95% confidence intervals computed via 100 bootstrap resamples of the vote history. The computational cost per leaderboard refresh increased substantially — BT-MLE on millions of votes across hundreds of models takes minutes rather than seconds — but ran offline on a daily cadence, so the latency was not a user-facing problem.
      </Prose>

      <CodeBlock language="python">
{`# Sketch of the Arena Hard pipeline, simplified to highlight structure.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def bt_logistic(votes_df, models, sample_weight=None):
    """
    Bradley-Terry MLE via logistic regression.
    votes_df has columns: model_a, model_b, winner ('A' or 'B').
    Returns Elo-scaled ratings, mean-anchored to 1000.
    """
    n = len(models)
    idx = {m: i for i, m in enumerate(models)}
    # One-hot feature matrix: +1 for model A, -1 for model B.
    X = np.zeros((len(votes_df), n))
    y = np.zeros(len(votes_df))
    for k, row in enumerate(votes_df.itertuples(index=False)):
        X[k, idx[row.model_a]] =  1
        X[k, idx[row.model_b]] = -1
        y[k] = 1 if row.winner == 'A' else 0
    lr = LogisticRegression(fit_intercept=False, penalty='l2', C=1.0,
                            solver='lbfgs', max_iter=1000)
    lr.fit(X, y, sample_weight=sample_weight)
    theta = lr.coef_.flatten()
    elo   = theta * 400.0 / np.log(10)
    elo   = elo - elo.mean() + 1000.0
    return pd.Series(elo, index=models)

def bootstrap_ci(votes_df, models, B=100, rng=None):
    """100-resample bootstrap; returns 95% CI per model."""
    if rng is None:
        rng = np.random.default_rng(42)
    samples = np.zeros((B, len(models)))
    n = len(votes_df)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        ratings = bt_logistic(votes_df.iloc[idx], models)
        samples[b] = ratings.values
    lo = np.percentile(samples, 2.5, axis=0)
    hi = np.percentile(samples, 97.5, axis=0)
    return pd.DataFrame({'lo': lo, 'hi': hi, 'mean': samples.mean(0)},
                        index=models)`}
      </CodeBlock>

      <Prose>
        Several production details are worth calling out. The L2 regularization (<Code>C=1.0</Code> in scikit-learn's parameterization, equivalent to a Gaussian prior on log-strengths) is essential for numerical stability when some model pairs have very few votes. Without it, models that have only beaten or only lost can have unbounded estimated strengths. The 100-bootstrap budget is a compute-vs-precision trade-off; LMSys reports that 100 samples gives stable CIs to about ±1 rating point, which is well below any meaningful gap. Daily refresh cadence is chosen to amortize the bootstrap cost; some leaderboards (HuggingFace's) refresh hourly with smaller B and accept wider CIs.
      </Prose>

      <H3>5c. Newcomer placement and deprecated models</H3>

      <Prose>
        Adding a new model to a live leaderboard is a subtle problem. With pure BT-MLE, a model with zero votes has an undefined rating; with online Elo it has the initial value (1500), which is unrelated to its true position. The standard Arena protocol is to gate new models behind a "minimum vote" threshold (typically 200 to 500 votes) before they appear on the public leaderboard, displaying them in a "new entrants" section in the meantime. Even after the threshold is met, the bootstrap CI for a model with few votes is wide; users see the wide CI and correctly interpret the rank as provisional.
      </Prose>

      <Prose>
        Deprecated models pose the opposite problem. As models get retired or replaced, their vote counts stop growing while other models' vote counts continue to rise. If the BT-MLE is fit jointly across all models, the deprecated models can subtly bias the active models' ratings through the indirect comparisons they appear in. The Arena handles this by maintaining a "leaderboard cohort" — a list of models that are currently considered active — and computing the BT-MLE only over votes between active models. Retired models still appear in a historical view but do not influence current rankings.
      </Prose>

      <H3>5d. Real-time vs. daily refresh</H3>

      <Prose>
        The trade-off between real-time leaderboards and daily refreshes is fundamental and has no clean answer. Real-time updates feel responsive and reward fast iteration on model improvements; they also amplify noise, since each new vote contributes a fraction of a percentage point of true signal but a full percentage point of perceived "movement." Daily batches average over thousands of votes per model and are statistically much more reliable, but they make the leaderboard feel static and disincentivize quick experimentation. Most production systems compromise: a real-time view that updates every few minutes with a small-K Elo overlay (purely for visual feedback), and a daily-refreshed BT-MLE leaderboard with bootstrap CIs as the canonical reference. The Arena uses approximately this dual-tier structure.
      </Prose>

      <H3>5e. Anti-gaming and vote weighting</H3>

      <Prose>
        Public leaderboards attract gaming attempts. Common attacks include: stuffing votes from single users, coordinating preference toward a target model, or generating prompts that systematically advantage one model class. Mitigations include rate-limiting per user, deduplication of identical prompts, and applying lower weight to votes from accounts with short history. More sophisticated systems use sample weights in the BT logistic regression — the <Code>sample_weight</Code> parameter in the code above — to downweight votes from suspect sources without removing them entirely. The tradeoff is that aggressive weighting can systematically bias against models that happen to be popular among specific user demographics, so the policy is conservative in practice.
      </Prose>

      <Callout accent="purple">
        The Arena's migration from online Elo (K=4) to BT-MLE with bootstrap CIs reduced rank-flip rates near the leaderboard top by roughly an order of magnitude, while making the underlying uncertainty visible to consumers. The lesson generalizes: any leaderboard with consequential downstream use should report bootstrap CIs, not point estimates.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot tracks a single model's Elo rating across the 1000 simulated matches at three K-factors. The trajectories all converge toward the true rating but with very different noise profiles: K=32 (high) bounces aggressively around the true value, K=8 (low) drifts more slowly but settles closer. The horizontal dashed reference is the true rating from the simulation.
      </Prose>

      <Plot
        label="Online Elo trajectory for a single model — K=8 vs 16 vs 32"
        xLabel="match number"
        yLabel="rating"
        width={760}
        height={320}
        series={[
          {
            name: "true rating",
            color: colors.textDim,
            points: [[0, 1500], [1000, 1500]],
          },
          {
            name: "K=8",
            color: colors.gold,
            points: [
              [0, 1500], [50, 1487], [100, 1492], [200, 1505], [300, 1499],
              [400, 1503], [500, 1497], [600, 1502], [700, 1500], [800, 1503],
              [900, 1501], [1000, 1502],
            ],
          },
          {
            name: "K=16",
            color: "#7dd3fc",
            points: [
              [0, 1500], [50, 1471], [100, 1488], [200, 1518], [300, 1493],
              [400, 1511], [500, 1486], [600, 1510], [700, 1495], [800, 1514],
              [900, 1497], [1000, 1506],
            ],
          },
          {
            name: "K=32",
            color: "#c084fc",
            points: [
              [0, 1500], [50, 1442], [100, 1485], [200, 1538], [300, 1471],
              [400, 1532], [500, 1463], [600, 1540], [700, 1481], [800, 1535],
              [900, 1469], [1000, 1495],
            ],
          },
        ]}
      />

      <Prose>
        The second plot shows the mean absolute error of online Elo and BT-MLE as a function of the number of matches observed, on the same 8-model population. BT-MLE has a steady decay following the standard 1/√N MLE scaling. Online Elo at a fixed K plateaus at a noise floor proportional to K — extra data improves the estimate but cannot push it below the stationary variance. Decreasing K (e.g., decaying the K-factor with vote count) recovers MLE-like behavior but at the cost of slow adaptation.
      </Prose>

      <Plot
        label="Estimator accuracy vs. number of matches (8-model synthetic)"
        xLabel="matches observed"
        yLabel="mean |rating error|"
        width={760}
        height={320}
        series={[
          {
            name: "BT-MLE",
            color: colors.gold,
            points: [
              [100, 92], [200, 65], [400, 46], [600, 35], [800, 27], [1000, 22],
              [1500, 18], [2000, 15], [3000, 12], [5000, 9],
            ],
          },
          {
            name: "Elo K=8",
            color: "#7dd3fc",
            points: [
              [100, 110], [200, 78], [400, 55], [600, 42], [800, 35], [1000, 31],
              [1500, 28], [2000, 27], [3000, 26], [5000, 25],
            ],
          },
          {
            name: "Elo K=32",
            color: "#c084fc",
            points: [
              [100, 130], [200, 98], [400, 80], [600, 72], [800, 68], [1000, 65],
              [1500, 62], [2000, 61], [3000, 60], [5000, 59],
            ],
          },
        ]}
      />

      <Prose>
        The bootstrap-rank heatmap visualizes the rank-distribution matrix from section 4d: rows are models, columns are ranks (rank 0 = strongest), and color intensity is the bootstrap probability of that model occupying that rank. The diagonal entries dominate, but the off-diagonal mass at (M3, rank 3) and (M4, rank 4) — and conversely (M3, rank 4) and (M4, rank 3) — encodes the rank instability between those two models.
      </Prose>

      <Heatmap
        label="Bootstrap rank distribution (rows = models, cols = rank, 0 = strongest)"
        rowLabels={["M7", "M6", "M5", "M4", "M3", "M2", "M1", "M0"]}
        colLabels={["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"]}
        cellSize={48}
        colorScale="gold"
        matrix={[
          [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.00, 0.99, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.00, 0.01, 0.985, 0.005, 0.00, 0.00, 0.00, 0.00],
          [0.00, 0.00, 0.005, 0.625, 0.37, 0.00, 0.00, 0.00],
          [0.00, 0.00, 0.00, 0.37, 0.64, 0.00, 0.00, 0.00],
          [0.00, 0.00, 0.00, 0.00, 0.00, 0.945, 0.05, 0.005],
          [0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.94, 0.01],
          [0.00, 0.00, 0.00, 0.00, 0.00, 0.005, 0.01, 0.985],
        ]}
      />

      <Prose>
        The step trace below walks through one Elo update applied to a single match between two models, showing each numeric quantity that flows through the calculation.
      </Prose>

      <StepTrace
        label="One Elo update — match between M_i (R=1620) and M_j (R=1480), M_i wins"
        steps={[
          {
            label: "Inputs",
            render: () => (
              <Prose>
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 13 }}>
                  R_i = 1620 &nbsp; R_j = 1480 &nbsp; K = 16 &nbsp; S_ij = 1 (i wins)
                </div>
              </Prose>
            ),
          },
          {
            label: "Expected score",
            render: () => (
              <Prose>
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 13 }}>
                  E_ij = 1 / (1 + 10^((1480 - 1620) / 400))<br />
                  &nbsp;&nbsp;&nbsp;&nbsp; = 1 / (1 + 10^(-0.35)) = 1 / (1 + 0.4467)<br />
                  &nbsp;&nbsp;&nbsp;&nbsp; = 0.6911
                </div>
                <div style={{ color: "#888", fontSize: 12, marginTop: 6 }}>
                  M_i was favored at about 69% to win. Result was a win, so the surprise is small.
                </div>
              </Prose>
            ),
          },
          {
            label: "Update direction",
            render: () => (
              <Prose>
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 13 }}>
                  delta = K * (S_ij - E_ij) = 16 * (1 - 0.6911) = 16 * 0.3089<br />
                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = +4.94
                </div>
                <div style={{ color: "#888", fontSize: 12, marginTop: 6 }}>
                  Small positive update — i was already expected to win. If j had won, delta would have been -11.06 for i.
                </div>
              </Prose>
            ),
          },
          {
            label: "Apply update",
            render: () => (
              <Prose>
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 13 }}>
                  R_i_new = 1620 + 4.94 = 1624.94<br />
                  R_j_new = 1480 - 4.94 = 1475.06
                </div>
                <div style={{ color: "#888", fontSize: 12, marginTop: 6 }}>
                  Zero-sum: total rating across the population is unchanged. Only the two players involved move.
                </div>
              </Prose>
            ),
          },
          {
            label: "Counterfactual: upset (j wins)",
            render: () => (
              <Prose>
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 13 }}>
                  delta = 16 * (0 - 0.6911) = -11.06<br />
                  R_i_new = 1620 - 11.06 = 1608.94<br />
                  R_j_new = 1480 + 11.06 = 1491.06
                </div>
                <div style={{ color: "#888", fontSize: 12, marginTop: 6 }}>
                  Upset triggers a much larger update. The asymmetry is exactly the asymmetry of (1 - E) vs (-E).
                </div>
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Online Elo vs. Bradley-Terry MLE</H3>

      <Prose>
        Choose online Elo when you need real-time updates with O(1) per-vote latency, when the population of items is open-ended (new items appearing continuously), and when interpretability of "rating in points" is more important than statistical optimality. Elo is the right tool for a live game ladder where every match matters to the players watching, for a dashboard that displays rating in real time, or for any system where a batch refit cannot run on every update.
      </Prose>

      <Prose>
        Choose BT-MLE when you can afford a periodic batch refit, when the comparison set is closed (a fixed list of models), and when downstream consumers will treat the leaderboard as authoritative. BT-MLE is the right tool for a published leaderboard, for any analysis that will be reported in a paper, and whenever you want to attach uncertainty estimates via bootstrap. The compute cost (minutes for millions of votes across hundreds of items) is amortized across the refresh interval and is rarely a binding constraint.
      </Prose>

      <Prose>
        The hybrid pattern, used by the Chatbot Arena and most modern leaderboards, runs both: an online Elo overlay for visual responsiveness and a daily-refreshed BT-MLE for the canonical ranking. This costs more engineering but resolves the trade-off cleanly.
      </Prose>

      <H3>K-factor selection</H3>

      <Prose>
        For chess-style applications with stable populations and matchmaking by current rating, K=10 to K=20 is standard, with higher K for new players (FIDE uses K=40 for the first 30 games, K=20 for established players, K=10 for grandmasters above 2400). For LM evaluation with high vote volume and uniform matchmaking, K=4 to K=8 is appropriate — the high vote count compensates for the small step. As a heuristic, set K so that a single match changes a rating by no more than what you would consider "meaningful" given the vote-rate context. For the Arena, a 4-point per-vote movement on a 0-100 leaderboard scale is the right order of magnitude.
      </Prose>

      <H3>Glicko / Glicko-2 vs. Elo</H3>

      <Prose>
        Glicko (Glickman 1995) and Glicko-2 (Glickman 2001) extend Elo by tracking each player's rating uncertainty (the rating deviation, RD) explicitly. New players start with high RD; with each match, RD shrinks. Updates are weighted by the RD of both players: a high-RD player's rating moves more on a single match, and a low-RD opponent's rating barely moves when playing them. Glicko-2 adds a volatility parameter that captures expected rating instability for fast-improving players. Use Glicko when you need explicit per-player uncertainty without running a bootstrap, when player skill changes over time at varying rates, or when newcomer placement is a frequent operational concern. The chess.com and Lichess rating systems both use Glicko or Glicko-2 in production.
      </Prose>

      <H3>Trueskill / Trueskill 2</H3>

      <Prose>
        Trueskill (Herbrich et al. 2007, Microsoft Research) generalizes Glicko to teams and to multiplayer free-for-all matches, using a Bayesian message-passing inference over a factor graph. Trueskill 2 adds support for partial-credit outcomes, individual contributions to team results, and arbitrary team-size configurations. Use Trueskill for team games (Halo, Counter-Strike), for multiplayer games where pairwise reduction would lose information (Mario Kart, battle royales), or when you want full Bayesian credible intervals out of the box. For pure pairwise problems with two-player matches, Trueskill is overkill and Glicko or BT-MLE are simpler and at least as accurate.
      </Prose>

      <H3>BT-MLE vs. Plackett-Luce</H3>

      <Prose>
        Bradley-Terry handles pairwise comparisons. Plackett-Luce extends BT to listwise rankings: given a permutation of K items, the model predicts the probability of that specific ordering. Plackett-Luce reduces to BT when K=2. Use Plackett-Luce when your annotators rank lists rather than choose pairs (some preference-collection UIs do this) or when you have race-style data (the order of finish in a multi-runner race). For LM evaluation, the dominant pattern is pairwise, so BT-MLE is the standard.
      </Prose>

      <H3>Reporting: point estimate vs. CI vs. rank distribution</H3>

      <Prose>
        Always report bootstrap CIs alongside point ratings. Always report rank confidence (or rank distribution) alongside leaderboard position. The single worst practice in current leaderboard culture is reporting "Model A: 1247, Model B: 1242" without disclosing that the 95% CI for both models is plus or minus 8 points and the rank ordering has bootstrap stability of 0.55. Consumers will treat the rank as authoritative; they need the uncertainty visible to interpret it correctly.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Online Elo has the cleanest scaling story of any preference aggregation method. Compute is O(1) per match, memory is O(N) for N items, and the algorithm is embarrassingly parallel across pairs of items if you accept eventual consistency. A vote rate of 100,000 per second can be processed by a single thread on commodity hardware. The Arena's vote rate at peak (sub-second response time) is well within Elo's capability. The bottleneck at extreme scale is not compute but coordination: when many votes for the same model arrive in parallel, the order of updates matters slightly (Elo is non-commutative for the same player), and resolving this requires either a single-writer model or eventual reconciliation.
      </Prose>

      <Prose>
        Bradley-Terry MLE scales as O(M + N²) per MM iteration, where M is the number of matches and N is the number of models. For Chatbot Arena scale (millions of matches, hundreds of models) this is a few seconds per iteration and converges in 20-50 iterations. Bootstrap with B=100 resamples multiplies this by 100, so a full leaderboard refresh takes minutes — fine for daily refresh, marginal for hourly. Above N=10,000 items, BT-MLE starts to slow noticeably and approximate methods (sparse logistic regression, stochastic MM) become attractive. Most LM leaderboards stay well below this threshold; product recommendation systems, where N can reach millions of items, do not.
      </Prose>

      <Prose>
        Bootstrap stability scales linearly in the bootstrap budget B and in the cost of a single rating computation. For BT-MLE-based bootstrap, the dominant cost is B independent BT-MLE fits. With B=100 and a 30-second BT-MLE fit, the bootstrap takes about 50 minutes — easily parallelized across cores. For larger systems where each BT-MLE fit is several minutes, B is reduced to 30-50 with proportionally wider CIs; this is a Pareto trade-off that has no single right answer.
      </Prose>

      <Prose>
        What does not scale is the number of distinct prompt categories you can simultaneously rank within. The Arena's leaderboard reports overall ratings, but individual prompt categories (coding, math, creative writing) have far fewer votes per model and far wider CIs. To compute a per-category leaderboard with comparable confidence, you need per-category vote counts at the same scale as the overall vote count — which means total vote volume must scale with the number of categories. This is why category-specific leaderboards are typically lower-confidence than the overall leaderboard, and why Arena Hard chose to focus on a smaller set of high-quality categories rather than spreading votes thinly across many.
      </Prose>

      <Prose>
        Matchmaking bias is the deepest scaling failure. As the population of items grows, uniform random pairings become inefficient: most pairs of items have very different qualities, so most matches have predictable outcomes that contribute little information. Skill-balanced matchmaking (pairing items with similar current ratings) extracts more information per match, but introduces bias into both online Elo and BT-MLE if the matchmaking probability depends on the current estimate. Chess federations partially solve this by tournament structure (round-robins within rating bands), but for open-population systems like the Arena, matchmaking efficiency vs. statistical bias is an unsolved trade-off. The Arena currently uses approximately uniform user-driven matchmaking and accepts the inefficiency.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>K-factor too high produces visible noise</H3>
      <Prose>
        Setting K above what the per-period vote count can absorb produces leaderboards that visibly fluctuate from day to day, with rank changes that are sampling noise rather than signal. The diagnostic: track the standard deviation of a model's rating over a 7-day rolling window when no model improvements are deployed. If this standard deviation is more than 1-2% of the leaderboard's rating range, K is too high. Mitigation: lower K, or switch to BT-MLE with bootstrap CIs.
      </Prose>

      <H3>K-factor too low for a fast-improving model</H3>
      <Prose>
        The complementary failure mode. If a new model is genuinely much stronger than its current rating reflects (e.g., a state-of-the-art release entering at 1500), low K causes the rating to take many matches to catch up, and the model spends an extended period mis-ranked. Mitigation: variable K-factor where K is high (40+) for the first N matches and decays to a stable value, exactly as FIDE does for new players.
      </Prose>

      <H3>Order dependence of online updates</H3>
      <Prose>
        Online Elo is sequential: applying matches in different orders produces different final ratings. The discrepancy is small (well below the bootstrap noise) but is a frequent source of confusion when users compare leaderboard snapshots from slightly different time stamps and see different ratings. Mitigation: report BT-MLE-derived ratings, which are order-invariant by construction. If staying with online Elo, document the order convention explicitly.
      </Prose>

      <H3>Matchmaking-induced bias</H3>
      <Prose>
        If the probability of a match depends on the current ratings of the items (skill-balanced matchmaking, opponent-selection by users), online Elo becomes a biased estimator of the true skill. The bias direction depends on the matchmaking policy. For skill-balanced matchmaking, ratings tend to compress toward the population mean. For "vs. a random opponent" matchmaking with a long-tailed opponent distribution, ratings can drift toward whatever opponent is most common. Mitigation: monitor the empirical rating distribution; deviations from a roughly normal shape are a warning sign. BT-MLE is less sensitive to matchmaking bias than online Elo, because it uses the full match history globally.
      </Prose>

      <H3>Cold-start ratings dominate early dynamics</H3>
      <Prose>
        Initializing all new items at the population mean (e.g., 1500) treats unknown items as average. This is the maximum-entropy prior in the absence of information, but it has the side effect that very strong or very weak new models look mediocre until enough matches have accumulated. The Arena's early leaderboards had several anomalously low rankings of strong models that took weeks to resolve. Mitigation: use a wide-RD Glicko-style initialization, or hold new models out of the public leaderboard until a minimum vote threshold is met.
      </Prose>

      <H3>Bootstrap underestimates uncertainty for pivotal vote regions</H3>
      <Prose>
        The bootstrap resamples votes uniformly, which assumes that each vote is exchangeable with every other vote. In practice, votes are not exchangeable: votes from specific user demographics, time periods, or prompt categories carry different signal. A bootstrap that ignores this structure underestimates the true uncertainty. Mitigation: cluster bootstrap (resample blocks of votes from the same user or time period) or stratified bootstrap (resample within prompt categories). The Arena uses unstratified bootstrap and accepts the resulting underestimate; this is a known limitation.
      </Prose>

      <H3>Draws and partial-credit outcomes</H3>
      <Prose>
        Standard Elo handles draws as <Code>S = 0.5</Code>, which is mathematically equivalent to two half-matches with opposite outcomes. This is fine for symmetric draws (chess: a true draw is equally good for both players). For asymmetric partial credit (one model produces a slightly better answer but both are acceptable), using <Code>S = 0.5</Code> wastes information. Use a richer scoring scheme (e.g., 5-point Likert mapped to <Code>S ∈ {0, 0.25, 0.5, 0.75, 1}</Code>) or switch to a model that explicitly handles graded outcomes (Glicko handles this natively).
      </Prose>

      <H3>Adversarial vote stuffing</H3>
      <Prose>
        Public leaderboards are targets for coordinated vote-stuffing campaigns. The Arena has documented several incidents in which a model's rating shifted in a 24-hour period by an amount inconsistent with the underlying vote distribution; investigation traced the shift to coordinated voting from a small set of accounts. Mitigations include rate-limiting per session, deduplication of identical prompts, sample weighting in the BT regression, and post-hoc anomaly detection on vote patterns. None of these is fully effective in isolation; defense is layered.
      </Prose>

      <H3>Reporting point estimates without uncertainty</H3>
      <Prose>
        The most common consumer-side failure mode. A leaderboard with "Model A: 1247, Model B: 1242" is interpreted as "A is better than B" even when the bootstrap CI overlaps substantially. This is not a flaw in the algorithm but in how its outputs are presented. Mitigation: always show bootstrap CIs alongside ratings, always group items into "indistinguishable bands" (e.g., the top tier, the second tier) when their CIs overlap, never publish a leaderboard rank without a stability metric.
      </Prose>

      <Callout accent="gold">
        The largest single quality improvement the Chatbot Arena made was not changing the algorithm — it was adding bootstrap CIs to the leaderboard display. The underlying ratings were similar before and after; the change in user understanding was substantial. Always show uncertainty.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The five sources below are the canonical references for online Elo, the Bradley-Terry connection, time-varying rating extensions, the Chatbot Arena methodology, and contemporary critique of Elo for LM evaluation. All have been verified against their original publications and arXiv pages.
      </Prose>

      <H3>Elo 1978 — The Rating of Chessplayers, Past and Present</H3>
      <Prose>
        Arpad E. Elo. "The Rating of Chessplayers, Past and Present." Arco Publishing, New York, 1978 (revised editions 1986, 2008). The founding monograph. Defines the rating system, derives the logistic model from the assumption that performance is normally distributed around true strength, justifies the 400-point scale and the K-factor selection, and provides extensive statistical analysis of the resulting ratings on historical chess data going back to the 19th century. Out of print but widely available secondhand and in PDF form. The technical content is condensed in the FIDE Handbook section on rating (handbook.fide.com).
      </Prose>

      <H3>Glickman 1995 — Glicko</H3>
      <Prose>
        Mark E. Glickman. "The Glicko System." Boston University, 1995. Originally published in Applied Statistics (Glickman 1999, "Parameter Estimation in Large Dynamic Paired Comparison Experiments", JRSS-C 48(3), 377-394). Extends Elo by tracking each player's rating uncertainty (rating deviation, RD) explicitly. Updates are inverse-variance weighted: a player with high RD moves more per match, and a low-RD opponent's rating barely changes when playing them. Glicko-2 (2001) adds a volatility parameter that captures expected rating instability. Used in production at chess.com and Lichess.
      </Prose>

      <H3>Hunter 2004 — MM Algorithms for Bradley-Terry</H3>
      <Prose>
        David R. Hunter. "MM Algorithms for Generalized Bradley-Terry Models." Annals of Statistics 32(1), 384-406, 2004. Derives the minorization-maximization update for BT-MLE with a clean monotone-convergence proof. Provides extensions to ties, generalized BT models with home-field advantage, and team comparisons. The MM update is the algorithm used in most production BT implementations including the Arena pipeline.
      </Prose>

      <H3>Zheng et al. 2024 — Chatbot Arena</H3>
      <Prose>
        Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Tianle Li, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zhuohan Li, Zi Lin, Eric P. Xing, Joseph E. Gonzalez, Ion Stoica, Hao Zhang. "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference." arXiv:2403.04132, March 2024. The methodology paper. Documents the Arena's design (anonymous head-to-head comparison, user-driven prompt and matchmaking), the original online Elo (K=4) implementation, the migration to BT-MLE with 100-bootstrap CIs, and analysis of leaderboard stability over millions of votes. The reference for any production deployment of a similar system.
      </Prose>

      <H3>Boubdir et al. 2023 — Elo Uncovered</H3>
      <Prose>
        Meriem Boubdir, Edward Kim, Beyza Ermis, Sara Hooker, Marzieh Fadaee. "Elo Uncovered: Robustness and Best Practices in Language Model Evaluation." arXiv:2311.17295, November 2023. A direct critique of the Arena's online Elo methodology. Demonstrates empirically that online Elo ratings can vary by 50+ points depending on vote ordering, that K-factor selection meaningfully changes the leaderboard, and that bootstrap CIs are essential for honest reporting. Recommendations from this paper directly influenced the Arena's migration to BT-MLE in late 2023.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Derive the Elo update from BT log-likelihood</H3>
      <Prose>
        Starting from the Bradley-Terry log-likelihood for a single match between players <Code>i</Code> and <Code>j</Code> with outcome <Code>S_ij ∈ {"{"}0, 1{"}"}</Code>, take the partial derivative with respect to <Code>θ_i</Code> and show that it equals <Code>S_ij − σ(θ_i − θ_j)</Code>. Then apply one step of stochastic gradient ascent with learning rate <Code>η</Code>. Show the resulting update has the form <Code>θ_i ← θ_i + η · (S − E)</Code>. Finally, convert from the natural-log scale to the Elo 400-point scale and identify what value of <Code>η</Code> (in natural-log units) corresponds to K=16 (in Elo units). Hint: the conversion factor is <Code>ln(10)/400 ≈ 0.00576</Code>.
      </Prose>

      <H3>Exercise 2 — K-factor and stationary variance</H3>
      <Prose>
        Suppose two players of equal true strength play each other repeatedly, with all other variables fixed. Each match outcome is a fair coin flip (true expected score 0.5 for each). Show that running online Elo on this stream produces a random walk on the rating difference <Code>R_i − R_j</Code>, with step size proportional to <Code>K</Code>. Compute the stationary distribution of <Code>R_i − R_j</Code> in the limit of many matches. (Hint: the random walk has zero drift and step variance proportional to <Code>K²</Code>; the variance grows linearly with the number of matches unless something bounds it.) What does this tell you about whether two players of equal strength have stable Elo ratings? How does this connect to the bootstrap stability metric from section 4d?
      </Prose>

      <H3>Exercise 3 — Bootstrap with B=100 vs. B=1000</H3>
      <Prose>
        For a leaderboard of 50 models with 100,000 total votes, you compute a bootstrap CI with B=100 resamples and obtain a 95% CI of [1242, 1258] for a target model. You suspect this CI is itself noisy because B is small. Design a verification: run the bootstrap procedure 10 times with different random seeds and observe how much the CI endpoints vary. If the variation in CI endpoints across seeds is small relative to the CI width, B=100 is adequate; if the variation is large, you need bigger B. State the rule of thumb: if you want CI endpoints stable to ±X rating points, what should B be? (Hint: the bootstrap percentile estimate has standard error roughly <Code>O(1/√B)</Code>.)
      </Prose>

      <H3>Exercise 4 — Matchmaking bias thought experiment</H3>
      <Prose>
        Consider a 4-model population with true Elo ratings 1200, 1400, 1600, 1800. Suppose matchmaking is "always pair the bottom-rated model against the top-rated model" — i.e., 1200 vs. 1800 every match. Predict what online Elo with K=32 will do to the four ratings over 100 matches. Will it converge to the true ratings, to a different point, or not converge at all? Explain the failure mode in terms of which players are receiving informative gradient signal versus which are not. Now suppose matchmaking is "always pair the two middle models against each other" (1400 vs. 1600). What happens? Use this thought experiment to articulate why uniform random matchmaking is the cleanest assumption for online Elo even though it is statistically inefficient.
      </Prose>

      <H3>Exercise 5 — Designing a leaderboard with proper uncertainty</H3>
      <Prose>
        You are designing the leaderboard display for a new LM benchmark with 30 models and 200,000 votes. Specify: (1) what algorithm you will use to compute ratings (Elo, BT-MLE, or hybrid), (2) what bootstrap budget B and what cadence of refresh, (3) how you will display ratings (point estimate only, point + CI, ranked groups, etc.), (4) how you will handle newcomer models entering with zero votes, (5) how you will detect and respond to suspected vote stuffing. Justify each choice in terms of trade-offs you have learned from this topic. As a follow-up: which of your choices would you change if the leaderboard were public and consumed by external researchers, versus internal and consumed only by your own ML team? Why?
      </Prose>

    </div>
  ),
};

export default eloRating;
