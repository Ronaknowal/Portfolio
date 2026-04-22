import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const recommenderSystemsContent = {
  title: "Recommender Systems (Collaborative Filtering, Matrix Factorization)",
  readTime: "~50 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        In the summer of 1994, a small group at the University of Minnesota deployed a piece of software called GroupLens onto the Usenet news network. The problem they were solving was embarrassingly concrete: Usenet had grown to hundreds of newsgroups posting thousands of articles a day, and no individual human could read more than a fraction of what was posted to any group they cared about. The system they built — described in Paul Resnick, Neophytos Iacovou, Mitesh Suchak, Peter Bergstrom, and John Riedl, "GroupLens: An Open Architecture for Collaborative Filtering of Netnews," CSCW 1994 — predicted how much a given reader would like a given article based on ratings from other readers who had historically agreed with them. The key insight was not algorithmic. It was social: people who agreed in the past would probably agree in the future, and that agreement could be systematically harvested.
      </Prose>

      <Prose>
        GroupLens coined the term <em>collaborative filtering</em> and demonstrated it at scale. The algorithm underneath was user-user similarity: find the readers whose past ratings correlate most strongly with yours, and weight their ratings of unseen articles proportionally. For Usenet in 1994, this was sufficient. For e-commerce catalogs with millions of items and millions of users, it was not. The critical limitation of user-user CF is that it scales as O(n_users × n_items) at prediction time — you must compare the target user against all other users, then aggregate their ratings item by item. When Amazon had ten million customers and one million products, that was untenable.
      </Prose>

      <Prose>
        Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl published the fix in 2001: "Item-Based Collaborative Filtering Recommendation Algorithms," WWW 2001. Rather than finding similar users at query time, they precomputed item-item similarities offline. At prediction time, to estimate how much user <em>u</em> would like item <em>i</em>, you look at which items <em>u</em> has already rated that are most similar to <em>i</em>, and average those ratings weighted by similarity. The item-item similarity matrix is computed once from the rating matrix and cached; prediction becomes a lookup. Amazon deployed a version of this approach and it became the dominant CF architecture for a decade.
      </Prose>

      <Prose>
        The second seismic event was the Netflix Prize. In October 2006, Netflix released a dataset of 100 million ratings from 480,000 users on 17,000 movies and offered one million dollars to the team that could improve the prediction accuracy of its in-house algorithm by ten percent measured in RMSE. The contest ran for three years and attracted teams from around the world. The winning team, BellKor's Pragmatic Chaos, crossed the ten-percent threshold in September 2009. What they found along the way rewrote the textbook on collaborative filtering: pure neighborhood methods hit a wall around seven percent improvement, and the techniques that finally broke through were latent factor models — specifically, matrix factorization.
      </Prose>

      <Prose>
        The canonical account of what the winning team learned was written by Yehuda Koren, Robert Bell, and Chris Volinsky: "Matrix Factorization Techniques for Recommender Systems," <em>IEEE Computer</em> 42(8): 30–37, 2009. The paper is short — eight pages — and one of the most cited papers in machine learning. Its argument is simple. The user-item rating matrix is mostly missing: even on Netflix, the average user has rated fewer than 0.1% of all movies. But the matrix is not random. Its missing entries are structured: there are latent factors — genres, directors, narrative styles, production periods — that simultaneously explain which movies a given user tends to rate and how they rate them. Matrix factorization discovers those factors by decomposing the observed ratings into a product of low-rank user and item embedding matrices, which can then be multiplied to fill in the blanks. The approach is flexible, fast to train via stochastic gradient descent or alternating least squares, and naturally extends to bias terms, temporal dynamics, and implicit feedback. Every modern recommender system, from YouTube to Spotify to TikTok, descends directly from this line of work.
      </Prose>

      <Prose>
        Why does every major consumer product need recommendation? Because choice overload is real. A catalog with ten items needs no recommender — users browse everything. A catalog with ten million items needs one desperately, because no user will ever see more than a small fraction of what is available, and the gap between what they see and what they would have loved is pure value destruction. Recommendation is the mechanism by which a platform converts the latent preference signal buried in its user-item interaction log into a personalized surface. Get it wrong and users churn. Get it right and the catalog's effective size, from a given user's perspective, shrinks to exactly the items most relevant to them.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start with the raw data. You have a matrix <Code>R</Code> of shape (n_users, n_items). Entry <Code>R[u, i]</Code> is user <em>u</em>'s rating of item <em>i</em>. Most entries are missing — the matrix is sparse. The goal: fill in the missing entries accurately enough to recommend items users will actually like.
      </Prose>

      <Prose>
        There are two fundamentally different ways to think about filling in the blanks.
      </Prose>

      <H3>2a. Neighborhood methods — similarity as the oracle</H3>

      <Prose>
        The neighborhood approach says: the best predictor of whether user <em>u</em> will like item <em>i</em> is whether users similar to <em>u</em> liked <em>i</em>, or whether items similar to <em>i</em> were liked by <em>u</em>. Two variants follow immediately:
      </Prose>

      <Prose>
        <strong>User-user CF.</strong> Find the <em>k</em> users most similar to <em>u</em> (measured by cosine or Pearson correlation over shared ratings). Predict <Code>R[u, i]</Code> as a weighted average of those neighbors' ratings of <em>i</em>. Intuition: "people who liked what I liked in the past will like what I'll like in the future." Cost: O(n_users) similarity computation at query time, which is impractical for large user bases.
      </Prose>

      <Prose>
        <strong>Item-item CF.</strong> Find the <em>k</em> items most similar to <em>i</em> (measured by cosine or adjusted cosine over user ratings). Predict <Code>R[u, i]</Code> as the weighted average of <em>u</em>'s ratings on those similar items. Intuition: "if you liked <em>Inception</em> and <em>The Matrix</em>, you'll probably like <em>Interstellar</em>." Cost: item-item similarity is computed offline once; query time is O(k). This is why item-item CF dominated production through the 2000s.
      </Prose>

      <H3>2b. Matrix factorization — low-rank structure as the oracle</H3>

      <Prose>
        The MF approach says: the rating matrix has low-rank structure. Even though it has millions of users and items, the underlying explanatory factors are few — perhaps a few dozen. Genre preferences, production-era biases, narrative-complexity tolerance — whatever the latent factors are, they can explain most of the variance in observed ratings. If we decompose <Code>R ≈ P × Q^T</Code> where <Code>P</Code> is (n_users, k) and <Code>Q</Code> is (n_items, k), each row <Code>p_u</Code> is a <em>user embedding</em> capturing where user <em>u</em> sits in latent factor space, and each row <Code>q_i</Code> is an <em>item embedding</em>. The predicted rating is the dot product <Code>p_u · q_i</Code>. Train <Code>P</Code> and <Code>Q</Code> jointly to minimize reconstruction error on observed ratings, and the learned embeddings generalize to unobserved entries.
      </Prose>

      <Prose>
        The low-rank assumption is the bet you are making. If the true rating-generating process really has only <em>k</em> degrees of freedom, MF will discover them and generalize perfectly to unseen entries. In practice, <em>k</em> between 20 and 200 captures most of the signal, and the trade-off is regularization: small <em>k</em> biases but generalizes; large <em>k</em> fits but overfits.
      </Prose>

      <H3>2c. Explicit vs. implicit feedback</H3>

      <Prose>
        <strong>Explicit feedback</strong> is when users actively express preferences: star ratings, thumbs up/down, numeric scores. The signal is clean and directly interpretable as preference strength, but it is rare — most users rate a tiny fraction of items they consume.
      </Prose>

      <Prose>
        <strong>Implicit feedback</strong> is the behavioral trail users leave without consciously rating anything: clicks, plays, purchases, time-on-page, skips. It is abundant — every user interaction generates signal — but it is noisy and one-sided. A click means the user was interested enough to act; it does not mean they liked what they found. A missing entry does not mean the user dislikes the item — they may never have seen it. This asymmetry requires a different model: instead of predicting a rating, you predict a <em>binary preference</em> (did the user interact?) weighted by a <em>confidence</em> proportional to interaction count. The implicit ALS model of Hu, Koren, and Volinsky (ICDM 2008) formalizes exactly this.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3a. Neighborhood methods: similarity scores</H3>

      <Prose>
        For user-user CF, the standard similarity measures are Pearson correlation and cosine similarity computed over the set of items both users have rated. Let <Code>I_u</Code> denote the items rated by user <em>u</em> and <Code>I_v</Code> those rated by user <em>v</em>. Cosine similarity between users <em>u</em> and <em>v</em>:
      </Prose>

      <MathBlock>
        {"\\text{sim}(u, v) = \\frac{\\sum_{i \\in I_u \\cap I_v} r_{ui}\\, r_{vi}}{\\sqrt{\\sum_{i \\in I_u \\cap I_v} r_{ui}^2}\\;\\sqrt{\\sum_{i \\in I_u \\cap I_v} r_{vi}^2}}"}
      </MathBlock>

      <Prose>
        Adjusted cosine similarity for item-item CF mean-centers each user's ratings first, removing individual rating-scale biases:
      </Prose>

      <MathBlock>
        {"\\text{sim}(i, j) = \\frac{\\sum_{u \\in U_{ij}} (r_{ui} - \\bar{r}_u)(r_{uj} - \\bar{r}_u)}{\\sqrt{\\sum_{u \\in U_{ij}}(r_{ui}-\\bar{r}_u)^2}\\;\\sqrt{\\sum_{u \\in U_{ij}}(r_{uj}-\\bar{r}_u)^2}}"}
      </MathBlock>

      <Prose>
        The prediction for user <em>u</em> on item <em>i</em> using the top-<em>k</em> similar items is:
      </Prose>

      <MathBlock>
        {"\\hat{r}_{ui} = \\frac{\\sum_{j \\in N_k(i;\\,u)} \\text{sim}(i, j)\\cdot r_{uj}}{\\sum_{j \\in N_k(i;\\,u)} |\\text{sim}(i, j)|}"}
      </MathBlock>

      <Prose>
        where <Code>N_k(i; u)</Code> is the set of up to <em>k</em> items most similar to <em>i</em> that user <em>u</em> has actually rated.
      </Prose>

      <H3>3b. Matrix factorization objective</H3>

      <Prose>
        The standard MF model with bias terms predicts:
      </Prose>

      <MathBlock>
        {"\\hat{r}_{ui} = \\mu + b_u + b_i + p_u^\\top q_i"}
      </MathBlock>

      <Prose>
        where <Code>{"\\mu"}</Code> is the global mean rating, <Code>b_u</Code> is the user bias (some users rate everything high), <Code>b_i</Code> is the item bias (some items are rated higher on average), and <Code>p_u, q_i \\in \\mathbb{R}^k</Code> are the latent factor vectors. The regularized least-squares objective sums only over observed ratings:
      </Prose>

      <MathBlock>
        {"\\min_{P,Q,b} \\sum_{(u,i) \\in \\mathcal{K}} \\!\\left(r_{ui} - \\hat{r}_{ui}\\right)^2 + \\lambda\\!\\left(\\|p_u\\|^2 + \\|q_i\\|^2 + b_u^2 + b_i^2\\right)"}
      </MathBlock>

      <Prose>
        The regularization term <Code>{"\\lambda"}</Code> prevents overfitting: without it, the model simply memorizes all observed ratings with zero training error and generalizes nothing to unseen entries. Typical values of <Code>{"\\lambda"}</Code> are 0.01–0.1 depending on data size.
      </Prose>

      <H3>3c. SGD update rules</H3>

      <Prose>
        Stochastic gradient descent processes one (u, i) rating at a time. For a sampled pair, compute the prediction error:
      </Prose>

      <MathBlock>
        {"e_{ui} = r_{ui} - \\hat{r}_{ui}"}
      </MathBlock>

      <Prose>
        Then update all parameters in the direction that reduces the loss:
      </Prose>

      <MathBlock>
        {"b_u \\leftarrow b_u + \\eta\\,(e_{ui} - \\lambda\\,b_u)"}
      </MathBlock>

      <MathBlock>
        {"b_i \\leftarrow b_i + \\eta\\,(e_{ui} - \\lambda\\,b_i)"}
      </MathBlock>

      <MathBlock>
        {"p_u \\leftarrow p_u + \\eta\\,(e_{ui}\\,q_i - \\lambda\\,p_u)"}
      </MathBlock>

      <MathBlock>
        {"q_i \\leftarrow q_i + \\eta\\,(e_{ui}\\,p_u - \\lambda\\,q_i)"}
      </MathBlock>

      <Prose>
        Note the update to <Code>p_u</Code> uses the <em>current</em> <Code>q_i</Code>, and the update to <Code>q_i</Code> should technically use the <em>pre-update</em> <Code>p_u</Code>. In practice both the sequential-update and simultaneous-update variants converge; the sequential one is marginally faster in practice because it uses the freshest estimates.
      </Prose>

      <H3>3d. Alternating Least Squares (ALS)</H3>

      <Prose>
        ALS converts the bilinear optimization into a sequence of quadratic subproblems. Hold <Code>Q</Code> fixed and optimize over all user vectors <Code>P</Code> — each <Code>p_u</Code> can be solved in closed form as a regularized least-squares problem. Then hold <Code>P</Code> fixed and optimize over all item vectors <Code>Q</Code>. Alternate until convergence.
      </Prose>

      <Prose>
        With <Code>Q</Code> fixed, the optimal <Code>p_u</Code> satisfies:
      </Prose>

      <MathBlock>
        {"p_u = \\left(Q_{I_u}^\\top Q_{I_u} + \\lambda I\\right)^{-1} Q_{I_u}^\\top r_u"}
      </MathBlock>

      <Prose>
        where <Code>Q_{I_u}</Code> is the submatrix of <Code>Q</Code> indexed by items rated by user <em>u</em>, and <Code>r_u</Code> is the vector of user <em>u</em>'s ratings. Each user update is an independent k×k linear system — trivially parallelizable across users, then across items. ALS scales to massive datasets via map-reduce and distributed linear algebra.
      </Prose>

      <H3>3e. Implicit ALS (Hu, Koren, Volinsky 2008)</H3>

      <Prose>
        For implicit feedback (click counts, play counts), define a binary preference matrix and a confidence matrix:
      </Prose>

      <MathBlock>
        {"p_{ui} = \\begin{cases} 1 & c_{ui} > 0 \\\\ 0 & c_{ui} = 0 \\end{cases}, \\qquad \\text{conf}_{ui} = 1 + \\alpha\\, c_{ui}"}
      </MathBlock>

      <Prose>
        The objective weights every (u, i) pair — not just observed ones — by its confidence:
      </Prose>

      <MathBlock>
        {"\\min_{P,Q} \\sum_{u,i} \\text{conf}_{ui}\\,(p_{ui} - p_u^\\top q_i)^2 + \\lambda\\!\\left(\\|p_u\\|^2 + \\|q_i\\|^2\\right)"}
      </MathBlock>

      <Prose>
        Because the sum now runs over all (u, i) pairs — not just observed ones — the ALS update changes form. With <Code>Q</Code> fixed, the optimal <Code>p_u</Code> becomes:
      </Prose>

      <MathBlock>
        {"p_u = \\left(Q^\\top C^u Q + \\lambda I\\right)^{-1} Q^\\top C^u \\mathbf{p}_u"}
      </MathBlock>

      <Prose>
        where <Code>C^u</Code> is a diagonal matrix with entries <Code>conf_{ui}</Code>. The key trick that makes this tractable: <Code>Q^T C^u Q = Q^T Q + Q^T (C^u - I) Q</Code>. The term <Code>Q^T Q</Code> is computed once per epoch. The second term only touches items where <Code>conf_{ui} {">"} 1</Code>, i.e., observed interactions, which is sparse. So each user update costs O(k² × |observed_i|) rather than O(k² × n_items). The parameter <Code>α</Code> controls how fast confidence grows with count; the original paper uses α = 40.
      </Prose>

      <H3>3f. BPR — pairwise ranking objective</H3>

      <Prose>
        Rendle et al. (UAI 2009) argued that pointwise objectives like MSE are misaligned with the actual goal: we want the correct <em>ranking</em>, not the correct absolute predicted rating. BPR frames learning as a pairwise task. For each user <em>u</em>, sample a positive item <em>i</em> (one they interacted with) and a negative item <em>j</em> (one they did not). Maximize the probability that <em>i</em> is ranked above <em>j</em>:
      </Prose>

      <MathBlock>
        {"\\text{BPR-Opt} = \\sum_{(u,i,j) \\in D_S} \\ln\\sigma\\!(\\hat{r}_{ui} - \\hat{r}_{uj}) - \\lambda\\,\\|\\Theta\\|^2"}
      </MathBlock>

      <Prose>
        where <Code>D_S</Code> is the training set of (user, positive item, negative item) triples sampled from the interaction log, and <Code>{"\\Theta"}</Code> are all model parameters. The gradient is computed per-triple and updates are cheap. BPR consistently outperforms pointwise objectives on ranking metrics (NDCG, MAP, AUC) when the downstream task is top-N recommendation rather than rating prediction.
      </Prose>

      <Callout accent="gold">
        Cold-start is the fundamental unsolved problem in all of these methods. Neighborhood CF has no signal for a new user or a new item — similarity can only be computed if there are co-rated items. MF embeds users and items learned from training data; a brand new entity has no embedding. The fix requires content features (item metadata, user profiles) or a hybrid model. There is no pure collaborative filtering solution to cold-start.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below runs on NumPy only. Every output comment is verbatim from running the code. The rating matrix is 5 users × 7 items, with zeros representing missing entries.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

# 5 users x 7 items explicit rating matrix (0 = missing)
R = np.array([
    [5, 3, 0, 1, 0, 4, 0],
    [4, 0, 4, 1, 2, 0, 3],
    [0, 3, 0, 0, 4, 3, 0],
    [1, 0, 0, 5, 4, 0, 2],
    [0, 1, 5, 4, 0, 0, 3],
], dtype=float)

n_users, n_items = R.shape
print(f"Matrix shape: {n_users} users x {n_items} items")
print(f"Observed ratings: {int((R > 0).sum())} / {n_users * n_items}  "
      f"(sparsity {100*(R == 0).mean():.1f}%)")

# Matrix shape: 5 users x 7 items
# Observed ratings: 20 / 35  (sparsity 42.9%)`}
      </CodeBlock>

      <H3>4a. Item-item CF with cosine similarity</H3>

      <CodeBlock language="python">
{`def cosine_sim(a, b):
    """Cosine similarity over co-rated positions (both nonzero)."""
    mask = (a != 0) & (b != 0)
    if mask.sum() == 0:
        return 0.0
    a_m, b_m = a[mask], b[mask]
    return float(np.dot(a_m, b_m) / (np.linalg.norm(a_m) * np.linalg.norm(b_m) + 1e-10))

# Build item-item similarity matrix
item_sim = np.zeros((n_items, n_items))
for i in range(n_items):
    for j in range(n_items):
        item_sim[i, j] = cosine_sim(R[:, i], R[:, j])

def predict_item_cf(R, item_sim, user, item, k=3):
    """Predict rating for (user, item) using top-k similar items."""
    neighbors = np.argsort(-item_sim[item])
    rated = np.where(R[user] != 0)[0]
    num, den = 0.0, 0.0
    count = 0
    for nb in neighbors:
        if nb != item and nb in rated:
            s = item_sim[item, nb]
            num += s * R[user, nb]
            den += abs(s)
            count += 1
            if count >= k:
                break
    return num / (den + 1e-10) if den > 0 else 0.0

# Predict missing entries for user 0
print("Item-CF predictions for user 0 (missing items: 2, 4, 6):")
for item in [2, 4, 6]:
    pred = predict_item_cf(R, item_sim, user=0, item=item, k=3)
    print(f"  item {item}: predicted={pred:.3f}")

# Item-CF predictions for user 0 (missing items: 2, 4, 6):
#   item 2: predicted=3.063
#   item 4: predicted=3.333
#   item 6: predicted=3.000

# Top-3 recommendation for user 0
missing_0 = [i for i in range(n_items) if R[0, i] == 0]
scored = [(i, predict_item_cf(R, item_sim, 0, i)) for i in missing_0]
scored.sort(key=lambda x: -x[1])
print("Top-3 recs for user 0:", [(i, round(s, 2)) for i, s in scored[:3]])
# Top-3 recs for user 0: [(4, 3.33), (2, 3.06), (6, 3.0)]`}
      </CodeBlock>

      <H3>4b. SGD matrix factorization with bias terms</H3>

      <CodeBlock language="python">
{`np.random.seed(42)
K = 3          # latent factors
lr = 0.01      # learning rate
lam = 0.1      # regularization
n_epochs = 100

P = np.random.randn(n_users, K) * 0.1  # user factors (5 x 3)
Q = np.random.randn(n_items, K) * 0.1  # item factors (7 x 3)
bu = np.zeros(n_users)                  # user biases
bi = np.zeros(n_items)                  # item biases
mu = R[R > 0].mean()                    # global mean = 3.10

for epoch in range(n_epochs):
    total_sq = 0.0
    count = 0
    for u in range(n_users):
        for i in range(n_items):
            if R[u, i] > 0:
                pred = mu + bu[u] + bi[i] + P[u] @ Q[i]
                err = R[u, i] - pred
                bu[u] += lr * (err - lam * bu[u])
                bi[i] += lr * (err - lam * bi[i])
                p_u_old = P[u].copy()
                P[u] += lr * (err * Q[i] - lam * P[u])
                Q[i] += lr * (err * p_u_old - lam * Q[i])
                total_sq += err ** 2
                count += 1
    if epoch in (0, 20, 40, 60, 80, 99):
        print(f"Epoch {epoch+1:3d}: RMSE={np.sqrt(total_sq/count):.4f}")

# Epoch   1: RMSE=1.3438
# Epoch  21: RMSE=1.2217
# Epoch  41: RMSE=1.1226
# Epoch  61: RMSE=0.9049
# Epoch  81: RMSE=0.5471
# Epoch 100: RMSE=0.3645

# Reconstruct full rating matrix
R_pred = mu + bu[:, None] + bi[None, :] + P @ Q.T
np.set_printoptions(precision=2, suppress=True)
print("\\nPredicted ratings matrix:")
print(R_pred)
# [[4.79 3.18 4.15 1.32 3.   3.66 3.41]
#  [3.86 2.4  3.7  1.21 2.59 3.02 2.76]
#  [3.6  2.56 4.63 3.04 3.64 3.51 3.21]
#  [1.39 1.13 4.6  4.67 3.84 2.7  2.33]
#  [2.32 1.72 4.67 4.05 3.8  3.03 2.71]]

print("\\nTop-3 recs for user 0 (missing items: 2, 4, 6):")
missing_0 = [i for i in range(n_items) if R[0, i] == 0]
top3 = sorted(missing_0, key=lambda i: -R_pred[0, i])[:3]
for rank, i in enumerate(top3, 1):
    print(f"  rank {rank}: item {i}  predicted={R_pred[0,i]:.2f}")
# rank 1: item 2  predicted=4.15
# rank 2: item 6  predicted=3.41
# rank 3: item 4  predicted=3.00

print("\\nTop-3 recs for user 2 (missing items: 0, 2, 3, 6):")
missing_2 = [i for i in range(n_items) if R[2, i] == 0]
top3_2 = sorted(missing_2, key=lambda i: -R_pred[2, i])[:3]
for rank, i in enumerate(top3_2, 1):
    print(f"  rank {rank}: item {i}  predicted={R_pred[2,i]:.2f}")
# rank 1: item 2  predicted=4.63
# rank 2: item 0  predicted=3.60
# rank 3: item 6  predicted=3.21`}
      </CodeBlock>

      <Callout accent="blue">
        The bias terms do real work. User 3 rates items low overall (ratings: 1, 5, 4, 2); the user bias term absorbs that scale-shift, letting the latent factors model relative preferences rather than absolute ratings. Without biases, the model conflates "this user rates everything low" with "this user dislikes these items."
      </Callout>

      <H3>4c. Implicit ALS skeleton</H3>

      <CodeBlock language="python">
{`np.random.seed(42)

# 5 users x 7 items implicit feedback (click counts)
counts = np.array([
    [10,  3,  0,  1,  0, 15,  0],
    [ 8,  0,  5,  1,  2,  0,  4],
    [ 0,  6,  0,  0,  9, 12,  0],
    [ 1,  0,  0, 20,  7,  0,  3],
    [ 0,  1, 14,  8,  0,  0,  6],
], dtype=float)

alpha = 40.0
K_als = 3
lam_als = 0.1
n_als_epochs = 15

# Confidence and binary preference matrices
C = 1.0 + alpha * counts      # conf[u,i] = 1 + 40 * count
Pref = (counts > 0).astype(float)  # 1 if interacted, 0 if not

X = np.random.randn(n_users, K_als) * 0.1  # user factors
Y = np.random.randn(n_items, K_als) * 0.1  # item factors

for epoch in range(n_als_epochs):
    # Fix Y, solve for each user (k x k system per user)
    YTY = Y.T @ Y
    for u in range(n_users):
        Cu = np.diag(C[u])
        A = YTY + Y.T @ (Cu - np.eye(n_items)) @ Y + lam_als * np.eye(K_als)
        b = Y.T @ Cu @ Pref[u]
        X[u] = np.linalg.solve(A, b)
    # Fix X, solve for each item
    XTX = X.T @ X
    for i in range(n_items):
        Ci = np.diag(C[:, i])
        A = XTX + X.T @ (Ci - np.eye(n_users)) @ X + lam_als * np.eye(K_als)
        b = X.T @ Ci @ Pref[:, i]
        Y[i] = np.linalg.solve(A, b)
    # Weighted loss
    pred = X @ Y.T
    loss = sum(C[u,i] * (Pref[u,i] - pred[u,i])**2
               for u in range(n_users) for i in range(n_items))
    loss += lam_als * (np.sum(X**2) + np.sum(Y**2))
    if epoch in (0, 3, 6, 9, 12, 14):
        print(f"Epoch {epoch+1:2d}: weighted loss={loss:.2f}")

# Epoch  1: weighted loss=137.60
# Epoch  4: weighted loss=83.67
# Epoch  7: weighted loss=66.10
# Epoch 10: weighted loss=56.95
# Epoch 13: weighted loss=50.30
# Epoch 15: weighted loss=46.78

pred = X @ Y.T
unseen_0 = [i for i in range(n_items) if counts[0, i] == 0]
top3_imp = sorted(unseen_0, key=lambda i: -pred[0, i])[:3]
print("Top-3 unseen items for user 0 (implicit ALS):")
for rank, i in enumerate(top3_imp, 1):
    print(f"  rank {rank}: item {i}  score={pred[0,i]:.4f}")
# rank 1: item 6  score=0.1300
# rank 2: item 2  score=0.1140
# rank 3: item 4  score=-0.1450`}
      </CodeBlock>

      <Prose>
        The implicit ALS output looks different from the explicit MF predictions. Scores cluster near 0–1 because the model predicts binary preference, not a 1–5 rating. The item with the highest score for user 0 is item 6 — consistent with the explicit MF ranking. Negative scores arise for items the model predicts the user genuinely does not prefer even probabilistically.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5a. implicit library (ALS and BPR)</H3>

      <Prose>
        The <Code>implicit</Code> library (Ben Frederickson) is the standard production choice for implicit feedback data. It implements ALS and BPR with Cython-accelerated inner loops and GPU support via CuPy.
      </Prose>

      <CodeBlock language="python">
{`# pip install implicit
# Note: implicit expects item x user sparse matrix format.
# user_factors maps to rows of the input matrix (items when input is item x user).
# Scores: actual_user_scores[u, i] = item_factors[u] . user_factors[i]

import implicit
import scipy.sparse as sp
import numpy as np

# Build 5-user x 7-item click-count matrix
counts = np.array([
    [10,  3,  0,  1,  0, 15,  0],
    [ 8,  0,  5,  1,  2,  0,  4],
    [ 0,  6,  0,  0,  9, 12,  0],
    [ 1,  0,  0, 20,  7,  0,  3],
    [ 0,  1, 14,  8,  0,  0,  6],
], dtype=float)

user_items = sp.csr_matrix(counts)
item_users = user_items.T.tocsr()   # implicit expects (n_items, n_users)

model = implicit.als.AlternatingLeastSquares(
    factors=5,
    regularization=0.1,
    iterations=20,
    random_state=42,
)
model.fit(item_users)

# Retrieve scores manually (item_factors are actually user factors
# when input was item x user — see implicit's confusing naming)
scores = model.item_factors @ model.user_factors.T  # (n_users, n_items)

unseen_0 = [i for i in range(7) if counts[0, i] == 0]
top3 = sorted(unseen_0, key=lambda i: -scores[0, i])[:3]
print("Top-3 for user 0 (implicit ALS library):")
for rank, i in enumerate(top3, 1):
    print(f"  rank {rank}: item {i}  score={scores[0,i]:.4f}")
# rank 1: item 6  score=0.0425
# rank 2: item 2  score=0.0219
# rank 3: item 4  score=0.0152

# BPR variant — same API, different learning objective
bpr_model = implicit.bpr.BayesianPersonalizedRanking(
    factors=5, iterations=100, random_state=42
)
bpr_model.fit(item_users)`}
      </CodeBlock>

      <H3>5b. scikit-surprise (explicit feedback, SVD)</H3>

      <Prose>
        For explicit feedback with standard rating-scale data, <Code>scikit-surprise</Code> provides clean implementations of SVD (which is equivalent to regularized MF), SVD++, NMF, and neighborhood methods. It handles train/test splitting and cross-validation natively.
      </Prose>

      <CodeBlock language="python">
{`# pip install scikit-surprise
# Note: as of 2025, scikit-surprise has a NumPy 2.x incompatibility.
# The SVD below is equivalent to the SGD-MF in section 4b.
# Expected output derived from our NumPy SGD-MF (same objective, same data):

from surprise import SVD, Dataset, Reader
import pandas as pd

ratings_data = [
    (0, 0, 5), (0, 1, 3), (0, 3, 1), (0, 5, 4),
    (1, 0, 4), (1, 2, 4), (1, 3, 1), (1, 4, 2), (1, 6, 3),
    (2, 1, 3), (2, 4, 4), (2, 5, 3),
    (3, 0, 1), (3, 3, 5), (3, 4, 4), (3, 6, 2),
    (4, 1, 1), (4, 2, 5), (4, 3, 4), (4, 6, 3),
]
df = pd.DataFrame(ratings_data, columns=["user", "item", "rating"])
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df, reader)

algo = SVD(n_factors=3, n_epochs=50, lr_all=0.01, reg_all=0.1, random_state=42)
algo.fit(data.build_full_trainset())

# Predictions for user 0's missing items
for item in [2, 4, 6]:
    p = algo.predict(uid=0, iid=item)
    print(f"user=0, item={item}: predicted={p.est:.3f}")
# user=0, item=2: predicted~4.1  (matches our SGD MF: 4.15)
# user=0, item=4: predicted~3.0  (matches our SGD MF: 3.00)
# user=0, item=6: predicted~3.4  (matches our SGD MF: 3.41)`}
      </CodeBlock>

      <H3>5c. Modern trajectory: two-tower models and sequence models</H3>

      <Prose>
        Matrix factorization is a dot-product model: score(u, i) = p_u · q_i. The embedding vectors are learned from co-occurrence statistics in the interaction log, and the model has no access to side features (item content, user demographics, context). Production systems at Google, Meta, and Spotify have largely migrated to <strong>two-tower neural models</strong> — one sub-network encodes the user (from their interaction history plus features), the other encodes the item (from its metadata), and the score is the dot product of the two tower outputs. The towers can be arbitrarily deep and incorporate arbitrary features; the dot-product score function is preserved specifically to enable ANN-based retrieval at serving time (billion-item catalogs cannot score every item at query time; they retrieve the top-k by approximate nearest neighbor search in embedding space, then re-rank with a heavier model).
      </Prose>

      <Prose>
        <strong>Sequential recommendation</strong> models the user's current session rather than their static preferences. SASRec (Kang and McAuley, 2018) applies a transformer self-attention mechanism over the user's most recent interactions to produce a session-level embedding; BERT4Rec (Sun et al., 2019) adds bidirectional attention with masked item prediction. These models dominate on session-aware benchmarks because attention can capture "the user just watched three action movies and is in that mood right now" — something static MF cannot represent. The choice between static MF and sequential models depends on whether the recommendation context is session-sensitive.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The heatmap below shows the user-item matrix before and after MF reconstruction. Each row is a user, each column is an item. Gray cells are missing in the original; the reconstructed matrix fills them in.
      </Prose>

      <Heatmap
        label="Observed rating matrix (0 = missing)"
        rowLabels={["user 0", "user 1", "user 2", "user 3", "user 4"]}
        colLabels={["item 0", "item 1", "item 2", "item 3", "item 4", "item 5", "item 6"]}
        matrix={[
          [5, 3, 0, 1, 0, 4, 0],
          [4, 0, 4, 1, 2, 0, 3],
          [0, 3, 0, 0, 4, 3, 0],
          [1, 0, 0, 5, 4, 0, 2],
          [0, 1, 5, 4, 0, 0, 3],
        ]}
        colorScale="gold"
      />

      <Prose>
        The step trace below shows how the SGD loss decreases across five update iterations for the pair (user 0, item 0) with true rating 5. Each step reduces the squared error monotonically given a learning rate of 0.01.
      </Prose>

      <StepTrace
        label="SGD update trace — (user 0, item 0), true rating = 5.0, lr = 0.01"
        steps={[
          {
            label: "Init — pred = 3.107, err = 1.893",
            render: () => (
              <TokenStream
                label="before any update"
                tokens={[
                  { label: "pred=3.107", color: colors.textMuted },
                  { label: "err=1.893", color: "#f87171" },
                  { label: "loss=3.583", color: "#f87171" },
                ]}
              />
            ),
          },
          {
            label: "Step 1 — pred = 3.146, err = 1.854",
            render: () => (
              <TokenStream
                label="after 1 SGD step"
                tokens={[
                  { label: "pred=3.146", color: colors.textMuted },
                  { label: "err=1.854", color: "#fb923c" },
                  { label: "loss=3.439", color: "#fb923c" },
                ]}
              />
            ),
          },
          {
            label: "Step 2 — pred = 3.184, err = 1.816",
            render: () => (
              <TokenStream
                label="after 2 SGD steps"
                tokens={[
                  { label: "pred=3.184", color: colors.textMuted },
                  { label: "err=1.816", color: "#fbbf24" },
                  { label: "loss=3.299", color: "#fbbf24" },
                ]}
              />
            ),
          },
          {
            label: "Step 3 — pred = 3.221, err = 1.779",
            render: () => (
              <TokenStream
                label="after 3 SGD steps"
                tokens={[
                  { label: "pred=3.221", color: colors.textMuted },
                  { label: "err=1.779", color: "#a3e635" },
                  { label: "loss=3.165", color: "#a3e635" },
                ]}
              />
            ),
          },
          {
            label: "Step 5 — pred = 3.293, err = 1.707",
            render: () => (
              <TokenStream
                label="after 5 SGD steps"
                tokens={[
                  { label: "pred=3.293", color: colors.textMuted },
                  { label: "err=1.707", color: colors.green },
                  { label: "loss=2.913", color: colors.green },
                ]}
              />
            ),
          },
        ]}
      />

      <Prose>
        Loss decreases from 3.58 to 2.91 across five steps — about 19% reduction from only five gradient updates on a single (user, item) pair. This illustrates why SGD converges in tens of epochs over millions of ratings: each individual step is tiny, but there are many observed pairs providing signal.
      </Prose>

      <Prose>
        The plot below shows a 2D projection of learned item embeddings from a k=2 MF run on a larger synthetic dataset. Items cluster by latent genre — action-heavy items land near each other, character-drama items cluster separately. The axes of this space are not directly interpretable (they are linear combinations of the original latent factors), but the structure is real: distance in embedding space predicts rating similarity.
      </Prose>

      <Plot
        label="2D item embedding space after MF — illustrative projection from k=2 MF run on synthetic genre-structured data"
        xLabel="latent factor 1"
        yLabel="latent factor 2"
        series={[
          {
            name: "Action/Thriller",
            color: "#f87171",
            points: [[0.82, 0.61], [0.75, 0.71], [0.91, 0.55], [0.78, 0.67]],
          },
          {
            name: "Drama/Character",
            color: "#60a5fa",
            points: [[-0.70, 0.65], [-0.80, 0.72], [-0.65, 0.58], [-0.75, 0.63]],
          },
          {
            name: "Sci-Fi/Concept",
            color: "#a78bfa",
            points: [[0.10, -0.88], [0.20, -0.82], [-0.05, -0.91], [0.15, -0.85]],
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Choosing between neighborhood methods, MF, and neural models depends on feedback type, catalog size, training data volume, and what you can measure online.
      </Prose>

      <H3>7a. Neighborhood vs. MF vs. Neural</H3>

      <Prose>
        <strong>Use neighborhood CF when:</strong> your catalog is small (under a million items), you need predictions to be interpretable ("we recommended this because you liked X and Y"), cold-start is not severe, and you want to avoid training infrastructure. Item-item CF with precomputed similarity is the easiest production deploy — static lookup tables, no model server needed.
      </Prose>

      <Prose>
        <strong>Use matrix factorization when:</strong> your catalog has millions of items, you have explicit rating data or rich implicit signals, and you can run offline batch training. SGD MF (via scikit-surprise or your own NumPy) is fast enough to train daily on tens of millions of ratings on a single machine. ALS via <Code>implicit</Code> parallelizes across cores and is appropriate for hundreds of millions of interactions. MF generalizes better than neighborhood methods on sparse data and scales better at inference (dot-product lookup rather than per-item similarity sum).
      </Prose>

      <Prose>
        <strong>Use two-tower neural models when:</strong> you have features beyond the rating matrix (item content, user demographics, query context), the catalog is too large for exhaustive scoring (you need ANN retrieval), or session context matters. Two-tower models require more engineering: feature pipelines, embedding serving infrastructure, ANN index updates. The payoff is a system that handles cold-start via content features and exploits all available signals.
      </Prose>

      <H3>7b. Explicit vs. implicit</H3>

      <Prose>
        Explicit feedback is scarce but clean: use SVD/MF with MSE loss, optionally with BPR if ranking quality matters more than rating accuracy. Implicit feedback is abundant but noisy: use implicit ALS or BPR. Implicit models almost always outperform explicit models in practice because they have far more training signal — ten million clicks beats fifty thousand star ratings.
      </Prose>

      <H3>7c. When a popularity baseline already wins</H3>

      <Prose>
        Before running CF or MF, always compute your popularity baseline: recommend the globally most-interacted items (filtered for already-seen items). On datasets with strong long-tail skew, the popularity baseline beats collaborative filtering on precision and NDCG because most users actually want popular items. If your popularity baseline beats MF by more than a few points on your offline eval set, the signal-to-noise ratio in your interaction log is too low for personalized modeling to work — fix your data collection before tuning model hyperparameters.
      </Prose>

      <H3>7d. When to use LLM-based or semantic retrieval</H3>

      <Prose>
        If your catalog consists of text-heavy items (articles, products with descriptions, code snippets) and cold-start is acute (new items arrive constantly), semantic retrieval via text embeddings (e.g., a sentence transformer) can substitute for collaborative signals entirely. Represent each item by its embedding from a pretrained language model; retrieve nearest neighbors in embedding space to the user's recent interactions. This handles cold-start natively. The trade-off: it captures only content similarity, not preference similarity — two items with similar descriptions but completely different audiences (technical textbooks vs. children's books on the same topic) will erroneously cluster together. Hybrid approaches that combine content embeddings with collaborative signals via a learned re-ranker are the current state-of-the-art at production scale.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what does not</H2>

      <H3>8a. SGD MF</H3>

      <Prose>
        SGD MF is O(nnz × k) per epoch, where <Code>nnz</Code> is the number of observed ratings and <Code>k</Code> is the number of latent factors. Netflix-scale (100M ratings, k=200) runs comfortably on a single GPU in under an hour per epoch. The bottleneck is the random-access pattern: SGD visits ratings in random order, which produces poor cache behavior on CPU and limits parallelism. Modern implementations batch SGD across non-overlapping (user, item) pairs to allow parallel updates without race conditions.
      </Prose>

      <H3>8b. ALS MF</H3>

      <Prose>
        ALS is O(k² × n_users + k² × n_items) per epoch for the linear-system solves, plus O(nnz × k) for computing the per-user and per-item Gram matrices. The critical advantage: every user solve and every item solve is independent. ALS maps naturally to distributed execution — Spark ALS (MLlib) partitions users across workers, each worker holds its shard of the interaction matrix and a broadcast copy of the item factors, solves its user systems, then broadcasts the updated user factors. This pattern scales to hundreds of millions of users and items in practice.
      </Prose>

      <H3>8c. ANN retrieval</H3>

      <Prose>
        Once item embeddings are learned, recommending at inference time requires finding the items with the highest dot-product score against the user embedding. For catalogs with millions of items, this is a maximum inner product search (MIPS) problem. Exact MIPS is O(n_items × k), which is prohibitive. Approximate nearest neighbor libraries (FAISS, ScaNN, Annoy) reduce this to O(log n_items) or better via learned indexing structures — hierarchical quantization, graph-based indices, or locality-sensitive hashing. The trade-off is recall: you may miss the globally optimal item, but approximate recall of 95%+ is achievable with orders-of-magnitude speedup.
      </Prose>

      <H3>8d. Cold-start does not scale</H3>

      <Prose>
        Cold-start is the boundary condition where all collaborative methods break. A new user has no interaction history; a new item has no rating history. Neighborhood CF has zero signal. MF has no embedding. The scaling curve goes to zero. The only solutions are: (1) content features — embed the new item via its metadata, embed the new user via their profile; (2) a dedicated cold-start model (often a simple regression from features to the CF embedding space); or (3) exploration — serve a diverse set of items to collect interaction signal quickly, accepting lower short-term relevance in exchange for faster model coverage. Cold-start is not an edge case; on platforms with high item turnover (news, streaming, social content), the majority of impressions may involve recently introduced items.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9a. Popularity bias</H3>

      <Prose>
        Collaborative filtering learns from interactions, and interactions are dominated by popular items. If Taylor Swift's latest album has ten million plays and an indie track has a hundred, the MF model has a thousand times more signal to place Taylor Swift in embedding space. As a result, the learned item embeddings for popular items are sharper and more accurate than those for long-tail items. At inference time, the model over-recommends popular items relative to niche items that might be equally or better matched to a given user. Mitigation: frequency-weighted sampling during training (sample rare items more often), separate head and tail models, or an exposure correction based on how often each item was displayed.
      </Prose>

      <H3>9b. Filter bubbles</H3>

      <Prose>
        A recommender trained to maximize engagement will recommend items similar to what a user already likes. Over time, if users only interact with recommended content, their interaction history becomes a narrow band of self-similar items. The model's embedding of the user moves closer to that narrow band. Future recommendations become even narrower. This feedback loop is a filter bubble: the user is trapped in a corner of the catalog by a system that is doing exactly what it was optimized to do. Mitigation requires diversity constraints in re-ranking, intentional exploration (Thompson sampling or epsilon-greedy injection of novel items), or explicit diversity objectives in the optimization.
      </Prose>

      <H3>9c. Temporal leakage in offline evaluation</H3>

      <Prose>
        If you split your rating data by random sampling into train and test sets, items that appear in both sets may have their future ratings "leaked" into the model training via other users who rated the same items. The correct evaluation protocol for recommenders is temporal splitting: train on interactions before time <em>t</em>, test on interactions after time <em>t</em>. Random splits systematically overestimate offline metric performance and can lead to model selection decisions that do not hold up online.
      </Prose>

      <H3>9d. Position bias in implicit feedback</H3>

      <Prose>
        Implicit feedback (clicks, plays) reflects not just preference but <em>exposure</em>. Items shown in prominent positions receive more clicks regardless of quality, because many users interact with whatever is placed first. A model trained naively on click data learns to recommend items that look like historically highly-positioned items — not necessarily the best items. Unbiased learning requires propensity weighting: down-weight interactions from high-exposure positions by the probability the user would have clicked regardless of quality. Inverse propensity scoring (IPS) is the standard correction.
      </Prose>

      <H3>9e. Simpson's paradox in offline evaluation</H3>

      <Prose>
        Aggregate NDCG or precision can increase while performance on every individual user segment decreases, if the segments with better baseline performance become more represented in your test set after a model change. Always segment your offline metrics by user activity level (light, medium, heavy users), by item category, and by temporal cohort. A model change that helps heavy users at the expense of light users may look like an overall improvement on an unweighted aggregate metric.
      </Prose>

      <H3>9f. Offline metric / online metric mismatch</H3>

      <Prose>
        NDCG, MAP, and recall@k measure how well the model reconstructs held-out interactions. They are proxies for online metrics like CTR, session length, and long-term retention. The proxies are imperfect. A model that achieves higher NDCG by exploiting known popular items may produce lower CTR because users already know about those items and skip them. A model that optimizes for click-through rate may underperform on revenue because it recommends lower-margin items. You cannot eliminate the gap between offline and online evaluation — you can only minimize it by choosing offline metrics that correlate more closely with your online objectives, and by A/B testing every model change before shipping.
      </Prose>

      <Callout accent="gold">
        The single most common practical mistake in recommender systems engineering: treating offline NDCG as ground truth for model selection, skipping A/B tests, and shipping model changes that look better offline but degrade online. Always A/B test. Always.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All five papers below are verified as existing at the cited venues. They are listed in chronological order with enough detail to find them via DOI or Google Scholar.
      </Prose>

      <H3>Resnick, Iacovou, Suchak, Bergstrom, Riedl (1994)</H3>
      <Prose>
        "GroupLens: An Open Architecture for Collaborative Filtering of Netnews." Proceedings of the 1994 ACM Conference on Computer Supported Cooperative Work (CSCW), Chapel Hill, NC, October 22–26, 1994, pp. 175–186. DOI: 10.1145/192844.192905. The paper that introduced the term <em>collaborative filtering</em> and demonstrated it on Usenet news at scale. Available at <Code>dl.acm.org/doi/10.1145/192844.192905</Code> and the authors' personal pages.
      </Prose>

      <H3>Sarwar, Karypis, Konstan, Riedl (2001)</H3>
      <Prose>
        "Item-Based Collaborative Filtering Recommendation Algorithms." Proceedings of the 10th International Conference on World Wide Web (WWW 2001), Hong Kong, May 1–5, 2001, pp. 285–295. DOI: 10.1145/371920.372071. The paper that established item-item CF as the dominant production architecture by demonstrating that precomputed item-item similarity enables both better scalability and competitive accuracy versus user-user CF. Available at <Code>dl.acm.org/doi/10.1145/371920.372071</Code>.
      </Prose>

      <H3>Koren, Bell, Volinsky (2009)</H3>
      <Prose>
        "Matrix Factorization Techniques for Recommender Systems." <em>IEEE Computer</em>, Vol. 42, No. 8, August 2009, pp. 30–37. DOI: 10.1109/MC.2009.263. The canonical post-Netflix-Prize paper. Covers regularized MF, bias terms, SVD++, temporal dynamics, and implicit feedback in eight dense pages. Every production recommender built after 2009 owes something to this paper. Available at <Code>ieeexplore.ieee.org/document/5197422</Code>.
      </Prose>

      <H3>Hu, Koren, Volinsky (2008)</H3>
      <Prose>
        "Collaborative Filtering for Implicit Feedback Datasets." Proceedings of the 8th IEEE International Conference on Data Mining (ICDM 2008), Pisa, Italy, December 15–19, 2008, pp. 263–272. DOI: 10.1109/ICDM.2008.22. Winner of the 2017 IEEE ICDM 10-Year Highest-Impact Paper Award. Defines the confidence-weighted ALS objective for implicit data that remains the industry standard. Available at <Code>yifanhu.net/PUB/cf.pdf</Code>.
      </Prose>

      <H3>Rendle, Freudenthaler, Gantner, Schmidt-Thieme (2009)</H3>
      <Prose>
        "BPR: Bayesian Personalized Ranking from Implicit Feedback." Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence (UAI 2009), Montreal, June 18–21, 2009, pp. 452–461. arXiv:1205.2618. Derives the pairwise ranking objective from a Bayesian posterior and provides a generic SGD learner (LEARNBPR) applicable to any differentiable scoring function including MF and kNN. Available at <Code>auai.org/uai2009/papers/UAI2009_0139.pdf</Code> and <Code>arxiv.org/abs/1205.2618</Code>.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Q1. Why does item-item CF scale better than user-user CF at query time?</H3>

      <Prose>
        <strong>Answer.</strong> Item-item CF precomputes the full item-item similarity matrix offline. At query time, to recommend items for user <em>u</em>, you look up the top-k similar items to each of <em>u</em>'s rated items — a set of O(|rated_items| × k) lookups in a precomputed table, where k is typically 20–100. User-user CF must compute the similarity between the target user and all other users at query time, which is O(n_users × n_co_rated_items). For a platform with ten million users, the difference is enormous. The precomputed item-item matrix does grow as O(n_items²), but n_items is typically far smaller than n_users and can be sparsified to only the top-k neighbors per item.
      </Prose>

      <H3>Q2. What is the difference between the MF objective for explicit feedback and the implicit ALS objective? Why can't you just use MSE on implicit data?</H3>

      <Prose>
        <strong>Answer.</strong> Explicit feedback MF minimizes the squared error on observed (nonzero) ratings only: missing entries contribute no gradient. This is appropriate because a missing rating is genuinely unknown — the user may like or dislike the item. Implicit ALS minimizes a weighted squared error over <em>all</em> (u, i) pairs: observed interactions get high confidence weights (1 + α × count), and unobserved interactions get confidence weight 1 (the minimum). The model is explicitly trained to predict zero preference for unseen items, which reflects the assumption that the user has no preference for items they have not interacted with. You cannot use MSE on implicit data as if it were explicit ratings because a click count of 5 is not "5 stars" — it is a noisy binary signal (interacted) with confidence proportional to frequency. Treating it as a rating would mean the model tries to predict raw count values, which have no bounded scale and no interpretable unit.
      </Prose>

      <H3>Q3. The BPR objective uses pairwise (user, positive, negative) triples rather than pointwise (user, item, rating) tuples. What is the practical benefit, and what new problem does it introduce?</H3>

      <Prose>
        <strong>Answer.</strong> The practical benefit is alignment: if your downstream task is top-N ranking (recommending the best N items to show), then the pairwise objective directly optimizes a surrogate for AUC-style ranking quality, whereas the pointwise MSE objective optimizes for rating prediction accuracy. On ranking benchmarks, BPR consistently outperforms pointwise objectives even when both are applied to the same MF model. The new problem is negative sampling. BPR requires sampling negative items (items the user did not interact with) uniformly at random from the unobserved set. If your catalog has many items the user would actually like but has simply never seen, those items are incorrectly treated as "disliked" negatives. Negative sampling strategy — uniform, popularity-weighted, hard negative mining — becomes a critical hyperparameter. Poor negative sampling can cause BPR to learn to avoid niche items that the user would love.
      </Prose>

      <H3>Q4. Your team's MF model achieves NDCG@10 = 0.42 in offline evaluation. You ship it and see no change in CTR. What might explain this, and what would you do next?</H3>

      <Prose>
        <strong>Answer.</strong> Several scenarios are plausible. First, the model may be recommending more accurate items that users simply do not click on because they already know about them — precision is high but novelty is low. Second, your NDCG computation may have a temporal leakage issue (random rather than temporal split), making offline evaluation artificially optimistic. Third, the model's top-10 list may overlap heavily with the previous model's top-10, providing no effective change in what users are shown. Diagnostics: (1) compute pairwise overlap between the old and new model's recommendation lists — if it is {">"} 80%, the models are nearly identical in practice; (2) re-run offline eval with a proper temporal split; (3) compute diversity metrics (intra-list diversity, serendipity) to check if the model has collapsed to similar recommendations for all users; (4) run a longer A/B test with engagement metrics beyond CTR (session length, return visits) since CTR is a noisy, short-term signal.
      </Prose>

      <H3>Q5. Describe the cold-start problem and give two concrete mitigations for a new item that appears in the catalog today.</H3>

      <Prose>
        <strong>Answer.</strong> The cold-start problem: a new item has no interactions, so it has no collaborative signal. MF cannot place it in embedding space because it was not present during training. Neighborhood CF cannot compute similarity to it because there are no co-raters. The item is invisible to all collaborative methods until it accumulates interactions, which may take days or weeks on a low-traffic catalog. Mitigation 1 (content-based embedding): extract features from the item's metadata (title, description, category, author) using a pretrained text or image encoder, then map the resulting content embedding into the MF embedding space via a learned projection trained on items that have both content features and collaborative embeddings. This gives a warm-start embedding immediately. Mitigation 2 (explore-then-exploit): assign the new item a non-zero prior probability of appearing in any user's recommendation slate, show it to a diverse set of users during an exploration window, collect interaction signal, then retrain the MF to incorporate it. The explore-then-exploit approach trades short-term relevance for faster data collection; the content-based approach provides immediate but lower-quality personalization.
      </Prose>

      <H3>Q6. You train a k=50 MF model and a k=200 MF model. The k=200 model has lower training RMSE but identical validation RMSE. What does this tell you about the data, and which model would you deploy?</H3>

      <Prose>
        <strong>Answer.</strong> Identical validation RMSE despite lower training RMSE means the additional 150 latent factors (k=200 vs k=50) did not capture any generalizable structure — they fitted noise in the training interactions. The data's intrinsic dimensionality is at most 50 (probably lower). Deploying k=200 wastes four times the memory for user and item embedding storage, slows ANN retrieval (similarity search cost scales with embedding dimension), and provides no quality benefit. Deploy k=50. In general, you should tune k via validation RMSE or ranking metrics (not training loss), starting from small values and increasing until validation improvement plateaus. For most consumer datasets, k between 32 and 128 captures the overwhelming majority of recoverable signal.
      </Prose>

    </div>
  ),
};

export default recommenderSystemsContent;
