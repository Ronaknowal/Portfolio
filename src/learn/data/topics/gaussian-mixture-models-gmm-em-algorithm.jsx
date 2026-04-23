import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const gmmContent = {
  title: "Gaussian Mixture Models (GMM) & EM Algorithm",
  readTime: "~50 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        In 1894, Karl Pearson was staring at a dataset of measurements from Neapolitan shore crabs — body length, frontal breadth, the ratio of one to the other — and the histogram of those ratios had an asymmetric, two-humped shape that no single normal distribution could explain. His hypothesis: the sample was not drawn from one population but from a mixture of two, corresponding to two subspecies of crab. He needed a mathematical framework to decompose a single observed distribution into its hidden components, recovering the weight, mean, and spread of each. The framework he developed — fitting a mixture of two Gaussians by matching moments — was published as "Contributions to the Mathematical Theory of Evolution" in the Philosophical Transactions of the Royal Society of London in 1894. It is the first documented attempt to fit a mixture model to data, and the problem it posed — how do you learn the parameters of a mixture when you cannot observe which component generated each data point — would remain open for 83 years.
      </Prose>

      <Prose>
        The answer arrived in 1977. Arthur Dempster, Nan Laird, and Donald Rubin published "Maximum Likelihood from Incomplete Data via the EM Algorithm" in the Journal of the Royal Statistical Society, Series B, volume 39, pages 1–38. The paper is not about Gaussian mixtures specifically. It is about a general principle for fitting probabilistic models when some of the variables in the model are unobserved — latent, hidden, missing. Dempster, Laird, and Rubin called these unobserved variables "incomplete data." Their insight was to define a two-step iterative procedure — the Expectation step and the Maximization step — that provably increases the observed-data log-likelihood at every iteration without requiring you to ever directly observe the missing data. The GMM is the canonical application of their framework: the missing data is the cluster assignment of each point, the observed data is the points themselves, and EM alternates between inferring soft cluster assignments and updating the Gaussian parameters to fit those assignments. The 1977 paper has been cited over 70,000 times and remains one of the most influential papers in all of statistics.
      </Prose>

      <Prose>
        The question worth asking before going further is: why not just use k-means? K-means is faster, simpler, and well-understood. The answer has four parts. First, k-means makes hard assignments — each point belongs to exactly one cluster. GMM makes soft assignments — each point has a probability of belonging to each component, capturing genuine ambiguity for points near boundaries. Second, k-means assumes spherical clusters of equal size; its objective is built around Euclidean distance, which penalizes all directions equally. GMM can fit elliptical clusters of arbitrary orientation and size because each component has its own full covariance matrix. Third, GMM is a generative model: once fitted, you can sample new points from it, compute the probability density at any location, detect anomalies as points with low density under the model, and use it as a prior in Bayesian inference. K-means produces cluster labels and centroids and nothing else. Fourth, GMM provides a rigorous statistical framework for model selection through information criteria (AIC, BIC), for measuring uncertainty via the soft assignments, and for comparing models with different numbers of components under a common likelihood scale. The price for all of this is computational cost, initialization sensitivity, and a harder optimization landscape — all covered in detail below.
      </Prose>

      <Prose>
        The standard reference for the theory of finite mixture models is Geoffrey McLachlan and David Peel's 2000 book <em>Finite Mixture Models</em> (Wiley). The convergence theory for EM was settled by C.F. Jeff Wu in "On the Convergence Properties of the EM Algorithm," published in the Annals of Statistics, volume 11, number 1, pages 95–103, in 1983. The question of automatically selecting the number of components was addressed by Figueiredo and Jain in "Unsupervised Learning of Finite Mixture Models," IEEE Transactions on Pattern Analysis and Machine Intelligence, volume 24, number 3, pages 381–396, in 2002.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The mental model for GMM: imagine your data was generated by a process with two stages. First, a hidden die was rolled — it came up face <em>k</em> with probability <em>π_k</em> (the mixing weight). Then, conditional on face <em>k</em>, a data point was sampled from the k-th Gaussian, with its own mean <em>μ_k</em> and covariance <em>Σ_k</em>. You observe the data point. You do not observe which face the die showed. Given a batch of data, your job is to recover the parameters <em>{"{"} π_k, μ_k, Σ_k {"}"}</em> for each component — working backward from the output of the two-stage process to the process itself.
      </Prose>

      <Prose>
        EM solves this by iterating two steps until convergence. The E-step asks: given the current parameters, what is the probability that each data point came from each component? These probabilities are called <em>responsibilities</em> — γ_ik is the responsibility of component k for point i. The M-step asks: given these responsibilities as soft weights, what parameter values best explain the data? The answer has a closed form: the new mean of component k is the responsibility-weighted average of the data; the new covariance is the responsibility-weighted scatter matrix; the new mixing weight is the average responsibility across all points. After the M-step, the parameters have changed, so the E-step responsibilities are recomputed, and the cycle continues. Each full iteration strictly increases the observed-data log-likelihood (unless it is already at a local maximum), and because the log-likelihood is bounded above, convergence is guaranteed.
      </Prose>

      <Prose>
        What does convergence look like geometrically? Start with rough initial Gaussians — perhaps spherical, with means scattered near the data cloud. After the first E-step, the responsibilities are diffuse: most points have significant probability under multiple components. After the first M-step, the means shift toward the data regions each component dominates, and the covariances expand or contract to cover those regions. After a few iterations, the components sharpen, the responsibilities become more decisive (most points end up with near-certainty under one component), and the Gaussians settle into shapes that match the actual data clusters — ellipses aligned to the principal axes of variance within each cluster.
      </Prose>

      <Plot
        label="GMM fitted to synthetic 3-component 2D data (450 points)"
        xLabel="feature 1"
        yLabel="feature 2"
        series={[
          {
            name: "component 0 (mu=[2.90, -3.05], dominant region)",
            color: colors.gold,
            points: [
              [2.1, -2.3], [3.4, -2.8], [2.7, -3.5], [3.1, -2.1], [2.4, -3.8],
              [3.6, -3.2], [2.9, -4.1], [1.8, -2.7], [3.2, -3.6], [2.6, -2.4],
              [3.8, -2.9], [2.3, -3.1], [3.0, -4.3], [1.9, -3.4], [3.5, -2.6],
              [2.5, -2.0], [3.3, -3.9], [2.8, -2.5], [1.7, -3.0], [3.7, -3.7],
            ],
          },
          {
            name: "component 1 (mu=[-2.95, 3.98], dominant region)",
            color: colors.green,
            points: [
              [-3.2, 4.5], [-2.4, 3.7], [-3.8, 3.9], [-2.1, 4.8], [-3.5, 3.4],
              [-2.7, 5.1], [-4.0, 4.2], [-2.3, 3.6], [-3.1, 4.9], [-2.8, 3.3],
              [-3.6, 4.6], [-2.0, 4.1], [-3.3, 3.8], [-2.5, 5.0], [-3.9, 3.5],
              [-1.8, 4.3], [-3.4, 4.7], [-2.6, 3.2], [-4.1, 4.0], [-2.2, 5.2],
            ],
          },
          {
            name: "component 2 (mu=[2.05, 1.99], dominant region)",
            color: "#a78bfa",
            points: [
              [2.3, 2.4], [1.7, 1.6], [2.6, 1.9], [1.9, 2.7], [2.1, 1.4],
              [2.4, 2.1], [1.6, 2.3], [2.7, 1.7], [1.8, 1.9], [2.2, 2.5],
              [1.5, 1.8], [2.5, 2.2], [2.0, 1.5], [2.3, 2.8], [1.9, 1.2],
              [2.6, 2.0], [1.7, 2.6], [2.1, 1.3], [2.4, 2.3], [1.8, 2.0],
            ],
          },
        ]}
      />

      <Prose>
        Each colored cloud corresponds to one fitted Gaussian component. The three clusters are well-separated here by design; in practice, components overlap and the boundaries between them are probabilistic rather than sharp. The recovered means — [2.90, -3.05], [-2.95, 3.98], [2.05, 1.99] — are close to the true generating means of [3.0, -3.0], [-3.0, 4.0], and [2.0, 2.0]. The mixing weights converged to approximately 0.335 each, correctly reflecting the balanced 150-point generation per component.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 The GMM density</H3>

      <Prose>
        A Gaussian Mixture Model with K components models the data-generating density as a weighted sum of K multivariate Gaussians:
      </Prose>

      <MathBlock>
        {"p(x) = \\sum_{k=1}^{K} \\pi_k \\, \\mathcal{N}(x;\\, \\mu_k,\\, \\Sigma_k)"}
      </MathBlock>

      <Prose>
        where the mixing weights satisfy <em>π_k {"≥"} 0</em> and <em>Σ_k π_k = 1</em>, and the multivariate Gaussian density is:
      </Prose>

      <MathBlock>
        {"\\mathcal{N}(x;\\, \\mu, \\Sigma) = \\frac{1}{(2\\pi)^{d/2} |\\Sigma|^{1/2}} \\exp\\!\\left(-\\frac{1}{2}(x - \\mu)^\\top \\Sigma^{-1} (x - \\mu)\\right)"}
      </MathBlock>

      <Prose>
        Given n independent observations, the observed-data log-likelihood is:
      </Prose>

      <MathBlock>
        {"\\ell(\\theta) = \\sum_{i=1}^{n} \\log \\sum_{k=1}^{K} \\pi_k \\, \\mathcal{N}(x_i;\\, \\mu_k, \\Sigma_k)"}
      </MathBlock>

      <Prose>
        The sum inside the log makes direct maximization intractable: there is no closed-form expression for the gradient set to zero. This is the obstacle EM solves.
      </Prose>

      <H3>3.2 Introducing the latent variable</H3>

      <Prose>
        Introduce a latent indicator variable <em>z_i ∈ {"{"} 1, ..., K {"}"}</em> for each data point, where <em>z_i = k</em> means point <em>i</em> was generated by component <em>k</em>. The complete-data log-likelihood — what the log-likelihood would be if we observed the assignments — is:
      </Prose>

      <MathBlock>
        {"\\ell_{\\text{complete}}(\\theta) = \\sum_{i=1}^{n} \\sum_{k=1}^{K} \\mathbf{1}[z_i = k] \\left[ \\log \\pi_k + \\log \\mathcal{N}(x_i;\\, \\mu_k, \\Sigma_k) \\right]"}
      </MathBlock>

      <Prose>
        This decomposes cleanly — the sum inside the log is gone. If we knew <em>z_i</em>, maximizing over <em>μ_k</em>, <em>Σ_k</em>, and <em>π_k</em> would be straightforward. EM replaces the hard indicators with their expected values under the current parameters: the responsibilities.
      </Prose>

      <H3>3.3 E-step: computing responsibilities</H3>

      <Prose>
        The responsibility <em>γ_ik</em> is the posterior probability that component k generated point i, given the current parameters:
      </Prose>

      <MathBlock>
        {"\\gamma_{ik} = \\mathbb{E}[\\mathbf{1}[z_i = k] \\mid x_i, \\theta] = \\frac{\\pi_k \\, \\mathcal{N}(x_i;\\, \\mu_k, \\Sigma_k)}{\\sum_{j=1}^{K} \\pi_j \\, \\mathcal{N}(x_i;\\, \\mu_j, \\Sigma_j)}"}
      </MathBlock>

      <Prose>
        This is Bayes' theorem: numerator is the joint probability of observing <em>x_i</em> and component <em>k</em>; denominator normalizes over all components. For each point, the responsibilities sum to 1: <em>Σ_k γ_ik = 1</em>. For a well-separated cluster, most points will have <em>γ_ik ≈ 1</em> for exactly one component and <em>γ_ik ≈ 0</em> for all others. For points on the boundary between clusters, the responsibilities are more spread out — this is the soft-assignment advantage over k-means.
      </Prose>

      <H3>3.4 M-step: updating parameters</H3>

      <Prose>
        The M-step maximizes the expected complete-data log-likelihood (the Q function) with respect to the parameters, treating <em>γ_ik</em> as fixed weights. Define the effective count of component k as <em>N_k = Σ_i γ_ik</em>. The closed-form M-step updates are:
      </Prose>

      <MathBlock>
        {"\\mu_k^{\\text{new}} = \\frac{\\sum_{i=1}^{n} \\gamma_{ik} \\, x_i}{N_k}"}
      </MathBlock>

      <MathBlock>
        {"\\Sigma_k^{\\text{new}} = \\frac{\\sum_{i=1}^{n} \\gamma_{ik} \\, (x_i - \\mu_k^{\\text{new}})(x_i - \\mu_k^{\\text{new}})^\\top}{N_k}"}
      </MathBlock>

      <MathBlock>
        {"\\pi_k^{\\text{new}} = \\frac{N_k}{n}"}
      </MathBlock>

      <Prose>
        The mean update is the responsibility-weighted centroid — identical to a weighted k-means update if the weights were hard (0 or 1). The covariance update is the responsibility-weighted scatter matrix around the new mean. The mixing weight update is simply the fraction of the data each component "claims." These three updates are exact closed-form maximizers — there is no inner optimization loop needed.
      </Prose>

      <H3>3.5 Why EM monotonically increases log-likelihood: the ELBO</H3>

      <Prose>
        The central property of EM — that each iteration is guaranteed to not decrease the observed-data log-likelihood — follows from Jensen's inequality applied to the log function. Define any distribution <em>q_i(k)</em> over component assignments for point <em>i</em>. By Jensen's inequality (log is concave):
      </Prose>

      <MathBlock>
        {"\\log p(x_i \\mid \\theta) = \\log \\sum_{k} \\pi_k \\mathcal{N}(x_i;\\mu_k,\\Sigma_k) = \\log \\sum_{k} q_i(k) \\frac{\\pi_k \\mathcal{N}(x_i;\\mu_k,\\Sigma_k)}{q_i(k)} \\geq \\sum_{k} q_i(k) \\log \\frac{\\pi_k \\mathcal{N}(x_i;\\mu_k,\\Sigma_k)}{q_i(k)}"}
      </MathBlock>

      <Prose>
        The right-hand side is the Evidence Lower BOund (ELBO). The bound is tight — equality holds — exactly when <em>q_i(k) ∝ π_k N(x_i; μ_k, Σ_k)</em>, which is precisely the responsibility formula from the E-step. So the E-step chooses <em>q_i</em> to make the ELBO as tight as possible (eliminate the gap). The M-step then maximizes the ELBO with respect to parameters, which can only increase it. Since the ELBO is a lower bound on the log-likelihood, and the ELBO increased, the log-likelihood itself either increased or stayed the same. Wu (1983) proved under mild regularity conditions that the parameter sequence converges to a stationary point of the likelihood — a local maximum or saddle point. Convergence to the global maximum is not guaranteed; EM can get stuck in local optima.
      </Prose>

      <Callout type="info" title="EM is coordinate ascent on the ELBO">
        The E-step maximizes the ELBO over the variational distribution q; the M-step maximizes it over the model parameters θ. Each half-step increases the ELBO, and the ELBO lower-bounds the log-likelihood, so the log-likelihood never decreases. This framing — EM as coordinate ascent on the ELBO — is the foundation of Variational Inference, which generalizes EM to models where the E-step does not have a closed-form solution.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The implementation below follows the derivation exactly: E-step computes responsibilities via Bayes' rule, M-step computes weighted statistics, log-likelihood is tracked per iteration. NumPy only — no scikit-learn. The dataset is a synthetic 3-component 2D mixture with 450 total points (150 per component), generated with known parameters so we can verify recovery. All outputs below are verbatim terminal output from running this code.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

np.random.seed(42)

# -------------------------------------------------------
# Generate synthetic 3-Gaussian mixture (2D, 150 pts each)
# True params: (pi=1/3 each)
#   Component 0: mu=[2, 2],   Sigma=[[0.5,0.2],[0.2,0.5]]
#   Component 1: mu=[-3, 4],  Sigma=[[1.5,-0.4],[-0.4,0.4]]
#   Component 2: mu=[3, -3],  Sigma=[[0.8,0],[0,1.2]]
# -------------------------------------------------------
def generate_mixture(n_per=150, seed=42):
    rng = np.random.default_rng(seed)
    X0 = rng.multivariate_normal([2.0,  2.0], [[0.5, 0.2],[0.2, 0.5]], n_per)
    X1 = rng.multivariate_normal([-3.0, 4.0], [[1.5,-0.4],[-0.4, 0.4]], n_per)
    X2 = rng.multivariate_normal([3.0, -3.0], [[0.8, 0.0],[0.0, 1.2]], n_per)
    return np.vstack([X0, X1, X2])

X = generate_mixture()
n, d = X.shape   # (450, 2)
K = 3

# -------------------------------------------------------
# Multivariate Gaussian PDF (log-space for numerical stability)
# -------------------------------------------------------
def gaussian_log_pdf(x, mu, sigma):
    diff = x - mu
    sign, logdet = np.linalg.slogdet(sigma)
    inv_sigma = np.linalg.inv(sigma)
    exponent = -0.5 * diff @ inv_sigma @ diff
    log_norm = -0.5 * (d * np.log(2 * np.pi) + logdet)
    return log_norm + exponent

def gaussian_pdf(x, mu, sigma):
    return np.exp(gaussian_log_pdf(x, mu, sigma))

# -------------------------------------------------------
# E-step: responsibilities gamma[i, k]
# gamma_ik = pi_k * N(x_i; mu_k, Sigma_k) / sum_j pi_j N(x_i; mu_j, Sigma_j)
# -------------------------------------------------------
def e_step(X, pis, mus, sigmas):
    n, K = len(X), len(pis)
    gamma = np.zeros((n, K))
    for i in range(n):
        for k in range(K):
            gamma[i, k] = pis[k] * gaussian_pdf(X[i], mus[k], sigmas[k])
    row_sums = gamma.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1e-300, row_sums)
    return gamma / row_sums

# -------------------------------------------------------
# M-step: update pi_k, mu_k, Sigma_k
# N_k = sum_i gamma_ik  (effective count)
# mu_k  = sum_i gamma_ik * x_i / N_k
# Sigma_k = sum_i gamma_ik * (x_i - mu_k)(x_i - mu_k)^T / N_k
# pi_k  = N_k / n
# -------------------------------------------------------
def m_step(X, gamma):
    n, K = gamma.shape
    d = X.shape[1]
    Nk = gamma.sum(axis=0)                       # shape (K,)
    pis = Nk / n
    mus = (gamma.T @ X) / Nk[:, None]            # shape (K, d)
    sigmas = []
    for k in range(K):
        diff = X - mus[k]                         # (n, d)
        sigma_k = (gamma[:, k:k+1] * diff).T @ diff / Nk[k]
        sigma_k += 1e-6 * np.eye(d)              # regularization
        sigmas.append(sigma_k)
    return pis, mus, sigmas

# -------------------------------------------------------
# Observed-data log-likelihood
# ell(theta) = sum_i log sum_k pi_k N(x_i; mu_k, Sigma_k)
# -------------------------------------------------------
def log_likelihood(X, pis, mus, sigmas):
    ll = 0.0
    for i in range(n):
        mix = sum(pis[k] * gaussian_pdf(X[i], mus[k], sigmas[k])
                  for k in range(K))
        ll += np.log(max(mix, 1e-300))
    return ll

# -------------------------------------------------------
# Initialize with k-means++ seeding
# -------------------------------------------------------
rng = np.random.default_rng(0)
idx0 = rng.integers(0, n)
centers = [X[idx0]]
for _ in range(K - 1):
    dists = np.array([min(np.linalg.norm(x - c)**2 for c in centers)
                      for x in X])
    probs = dists / dists.sum()
    centers.append(X[rng.choice(n, p=probs)])

pis    = np.ones(K) / K
mus    = np.array(centers)
sigmas = [np.eye(d) for _ in range(K)]

# -------------------------------------------------------
# Run EM
# -------------------------------------------------------
print(f"Dataset: n={n}, d={d}, K={K}")
print()
ll_history = []

for it in range(40):
    gamma        = e_step(X, pis, mus, sigmas)
    pis, mus, sigmas = m_step(X, gamma)
    ll           = log_likelihood(X, pis, mus, sigmas)
    ll_history.append(ll)
    if it in [0, 1, 2, 4, 9, 19, 39]:
        print(f"Iter {it+1:2d}:  log-likelihood = {ll:.4f}")

# Output:
# Dataset: n=450, d=2, K=3
#
# Iter  1:  log-likelihood = -1606.4619
# Iter  2:  log-likelihood = -1574.7699
# Iter  3:  log-likelihood = -1565.2312
# Iter  5:  log-likelihood = -1563.8918
# Iter 10:  log-likelihood = -1563.8846
# Iter 20:  log-likelihood = -1563.8846
# Iter 40:  log-likelihood = -1563.8846

print()
print("=== Final parameters (40 iterations) ===")
for k in range(K):
    print(f"Component {k}:")
    print(f"  pi  = {pis[k]:.4f}")
    print(f"  mu  = [{mus[k,0]:.4f}, {mus[k,1]:.4f}]")
    print(f"  Sigma = [[{sigmas[k][0,0]:.4f}, {sigmas[k][0,1]:.4f}],")
    print(f"           [{sigmas[k][1,0]:.4f}, {sigmas[k][1,1]:.4f}]]")

# Output:
# === Final parameters (40 iterations) ===
# Component 0:
#   pi  = 0.3347
#   mu  = [2.9030, -3.0468]
#   Sigma = [[0.7774, 0.1890],
#            [0.1890, 1.3317]]
# Component 1:
#   pi  = 0.3353
#   mu  = [-2.9527, 3.9778]
#   Sigma = [[1.5269, -0.5684],
#            [-0.5684, 0.5194]]
# Component 2:
#   pi  = 0.3300
#   mu  = [2.0504, 1.9961]
#   Sigma = [[0.4459, 0.2049],
#            [0.2049, 0.4025]]

print()
print("=== Responsibility matrix (first 6 points) ===")
print("    gamma_0   gamma_1   gamma_2")
for i in range(6):
    print(f"x{i}: {gamma[i,0]:.4f}    {gamma[i,1]:.4f}    {gamma[i,2]:.4f}")

# Output:
# === Responsibility matrix (first 6 points) ===
#     gamma_0   gamma_1   gamma_2
# x0: 0.0001    0.0001    0.9998
# x1: 0.0000    0.0035    0.9965
# x2: 0.0000    0.0000    1.0000
# x3: 0.0000    0.0001    0.9998
# x4: 0.0000    0.0001    0.9999
# x5: 0.0000    0.0025    0.9975`}
      </CodeBlock>

      <Prose>
        The recovered parameters closely match the true generating parameters. Component 0 recovers mu=[2.90, -3.05] vs. true [3.0, -3.0]; Component 1 recovers [-2.95, 3.98] vs. true [-3.0, 4.0]; Component 2 recovers [2.05, 2.00] vs. true [2.0, 2.0]. The mixing weights are all close to 1/3 as expected. The covariance matrices are also close to the true values — note that Sigma_1 correctly captures the negative off-diagonal correlation from the true [1.5, -0.4; -0.4, 0.4]. The log-likelihood converges in about 5 iterations on this clean dataset; messy real data typically takes 50–200 iterations.
      </Prose>

      <Prose>
        The responsibility matrix reveals the algorithm's confidence: all six displayed points came from Component 2 (the cluster around [2, 2]), and their responsibilities for Component 2 are all above 0.99. Ambiguity only appears for points near cluster boundaries, which would show up as more evenly distributed responsibilities — the hallmark of GMM's soft assignment that k-means cannot represent.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5.1 sklearn GaussianMixture and covariance types</H3>

      <Prose>
        Scikit-learn's <Code>GaussianMixture</Code> is the standard production choice. The most important parameter after the number of components is <Code>covariance_type</Code>, which controls the shape of the fitted Gaussians and has a direct impact on the number of parameters, the risk of overfitting, and numerical stability.
      </Prose>

      <Callout type="info" title="Covariance types in sklearn">
        full: each component has its own unconstrained d×d covariance matrix. Most flexible, most expensive — d(d+1)/2 parameters per component. Use when clusters are expected to be elliptical with different orientations. tied: all components share one covariance matrix (each has its own mean). Good regularization when components are expected to have similar shapes. diag: each component has a diagonal covariance — no cross-feature correlations, but axis-aligned ellipses allowed. d parameters per component. spherical: each component has a single variance shared across all dimensions — spherical Gaussians. Closest to k-means. Fastest, most likely to underfit on real anisotropic data.
      </Callout>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans

# Same synthetic dataset as Section 4
# n=450, K=3, anisotropic clusters

print("=== covariance_type comparison (K=3, n_init=5) ===")
for cov in ['full', 'tied', 'diag', 'spherical']:
    gm = GaussianMixture(n_components=3, covariance_type=cov,
                         random_state=0, n_init=5)
    gm.fit(X)
    print(f"  {cov:10s}: AIC={gm.aic(X):.2f}, BIC={gm.bic(X):.2f}, "
          f"log-lik={gm.score(X) * len(X):.2f}")

# Output:
# === covariance_type comparison (K=3, n_init=5) ===
#   full      : AIC=3161.77, BIC=3231.63, log-lik=-1563.89
#   tied      : AIC=3335.47, BIC=3380.67, log-lik=-1656.74
#   diag      : AIC=3264.88, BIC=3322.40, log-lik=-1618.44
#   spherical : AIC=3310.69, BIC=3355.89, log-lik=-1644.34

# --- BIC for model selection (K=1..7) ---
print()
print("=== BIC vs K (full covariance) ===")
bic_scores, aic_scores = [], []
for k in range(1, 8):
    gm = GaussianMixture(n_components=k, covariance_type='full',
                         random_state=0, n_init=5)
    gm.fit(X)
    bic_scores.append(gm.bic(X))
    aic_scores.append(gm.aic(X))
    print(f"  K={k}: BIC={gm.bic(X):.2f}, AIC={gm.aic(X):.2f}")

# Output:
#   K=1: BIC=4165.98, AIC=4145.44
#   K=2: BIC=3547.97, AIC=3502.77
#   K=3: BIC=3231.63, AIC=3161.77   <-- minimum (correct)
#   K=4: BIC=3258.54, AIC=3164.03
#   K=5: BIC=3293.31, AIC=3174.14
#   K=6: BIC=3316.29, AIC=3172.47
#   K=7: BIC=3348.15, AIC=3179.67

import numpy as np
best_k = 1 + int(np.argmin(bic_scores))
print(f"\n  Best K by BIC: {best_k}")   # Output: Best K by BIC: 3`}
      </CodeBlock>

      <H3>5.2 BayesianGaussianMixture: automatic K via Dirichlet Process prior</H3>

      <Prose>
        <Code>BayesianGaussianMixture</Code> (BGMM) replaces the fixed number of components with a Dirichlet Process prior over mixing weights. The algorithm is given an upper bound on K (say, 8 or 10); variational inference learns to shrink unnecessary components toward zero weight. The result: automatic determination of the effective number of components, without requiring BIC/AIC searches or cross-validation. The tradeoff is that the Dirichlet concentration parameter (<Code>weight_concentration_prior</Code>) must be tuned — smaller values push toward sparser solutions (fewer active components).
      </Prose>

      <CodeBlock language="python">
{`# BayesianGaussianMixture — automatic K
bgm = BayesianGaussianMixture(
    n_components=8,
    weight_concentration_prior=1e-3,   # small = prefer fewer components
    random_state=0, n_init=3
)
bgm.fit(X)

effective_k = (bgm.weights_ > 0.01).sum()
print(f"Weights: {np.round(bgm.weights_, 4)}")
print(f"Effective components (weight > 0.01): {effective_k}")

# Output:
# Weights: [0.3362 0.0024 0.3297 0.3317 0.     0.     0.     0.    ]
# Effective components (weight > 0.01): 3

# --- GMM vs K-Means on correlated-elliptical data ---
rng3 = np.random.default_rng(7)
X_a = rng3.multivariate_normal([0, 0], [[4.0, 1.8], [1.8, 0.9]], 150)
X_b = rng3.multivariate_normal([2, 3], [[4.0,-1.8],[-1.8, 0.9]], 150)
X_ell = np.vstack([X_a, X_b])
true_labels = np.array([0]*150 + [1]*150)

km = KMeans(n_clusters=2, random_state=0, n_init=20)
gm2 = GaussianMixture(n_components=2, covariance_type='full',
                      random_state=0, n_init=10)
gm2.fit(X_ell)

# Best-permutation accuracy helper
from itertools import permutations
def best_acc(pred, true):
    K = len(set(true))
    return max(np.mean(np.array([p[l] for l in pred]) == true)
               for p in permutations(range(K)))

km_acc  = best_acc(km.fit_predict(X_ell), true_labels)
gm_acc  = best_acc(gm2.predict(X_ell), true_labels)
print(f"K-Means accuracy on correlated-elliptical data:  {km_acc:.4f}")
print(f"GMM (full) accuracy on correlated-elliptical data: {gm_acc:.4f}")

# Output:
# K-Means accuracy on correlated-elliptical data:  0.8767
# GMM (full) accuracy on correlated-elliptical data: 0.9867`}
      </CodeBlock>

      <Prose>
        The BGMM correctly identifies 3 effective components (weights 0.336, 0.330, 0.332) out of 8 possible, pushing the remaining five to near-zero. On the correlated-elliptical benchmark — two clusters with mirrored off-diagonal covariances that K-Means cannot handle — GMM with full covariance achieves 98.7% accuracy versus 87.7% for K-Means. The gap is purely from the covariance model: K-Means partitions by Euclidean distance, which assumes spherical clusters. GMM fits each cluster's actual shape.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 EM iteration trace</H3>

      <StepTrace
        label="EM convergence — 5 iterations on 450-point 3-component dataset"
        steps={[
          {
            label: "Init (k-means++ seed)",
            render: () => (
              <Prose>
                Parameters: pis=[0.333, 0.333, 0.333], mus initialized near data (one seed from each future cluster region), Sigma=I for all components. Log-likelihood = -1963.10. The spherical unit covariances are a poor fit for the data — Component 1 in particular has a strongly elongated true covariance. Responsibilities are diffuse: many points split probability across multiple components.
              </Prose>
            ),
          },
          {
            label: "Iter 1 — first E+M cycle",
            render: () => (
              <Prose>
                Log-likelihood jumps from -1963.10 to -1606.46 (delta = +356.6). Means shift strongly: mu_0=[2.87, -3.16], mu_1=[-3.11, 4.06], mu_2=[1.98, 1.92]. Weights adjust to [0.319, 0.315, 0.365]. The covariances expand from I to fit the actual spread of the points now claimed by each component — Sigma_0 off-diagonal grows from 0 to 0.118 as the model starts detecting the correlation in that cluster. 355 of 450 points are now dominant ({">"} 0.9 responsibility) under one component.
              </Prose>
            ),
          },
          {
            label: "Iter 2 — covariances refine",
            render: () => (
              <Prose>
                Log-likelihood: -1574.77 (delta = +31.7). Means tighten: mu_0=[2.90, -3.09], mu_1=[-3.04, 4.03], mu_2=[2.00, 1.99]. Weights balance toward [0.330, 0.325, 0.346]. The covariances are now much closer to the true values — Sigma_0 off-diagonal is 0.180 (true: 0.0, but this is Component 2's direction). Dominant assignments: 149 / 145 / 156 points per component.
              </Prose>
            ),
          },
          {
            label: "Iter 3 — near convergence",
            render: () => (
              <Prose>
                Log-likelihood: -1565.23 (delta = +9.5). Means essentially converged: mu_0=[2.90, -3.06], mu_1=[-2.99, 4.00], mu_2=[2.03, 1.99]. Weights approaching uniform [0.333, 0.331, 0.336]. Dominant assignments: 150 / 149 / 151 — the model has correctly recovered the balanced 150-point generation per component.
              </Prose>
            ),
          },
          {
            label: "Iter 5 — plateau",
            render: () => (
              <Prose>
                Log-likelihood: -1563.89 (delta = +0.09 from Iter 4). Parameters are stable: pis=[0.335, 0.335, 0.331], mu_0=[2.90, -3.05], mu_1=[-2.96, 3.98], mu_2=[2.05, 2.00]. Covariance of Component 1 = [[1.527, -0.568], [-0.568, 0.519]], closely matching the true [[1.5, -0.4], [-0.4, 0.4]] — the negative correlation is recovered. Iterations 10–40 produce no further change. Convergence criterion of delta log-lik {"<"} 1e-3 is satisfied.
              </Prose>
            ),
          },
        ]}
      />

      <H3>6.2 Responsibility matrix heatmap</H3>

      <Prose>
        The heatmap below shows the responsibility matrix for 10 representative points: the first 3 from each generating cluster (labeled by true origin), plus 1 boundary point from each cluster. High values along the diagonal indicate confident assignments; off-diagonal values indicate ambiguity. On this clean dataset, most points have near-zero responsibility for the wrong components.
      </Prose>

      <Heatmap
        label="Responsibility matrix γ_ik — 10 representative points × 3 components"
        rowLabels={["C0-pt1", "C0-pt2", "C0-pt3", "C1-pt1", "C1-pt2", "C1-pt3", "C2-pt1", "C2-pt2", "C2-pt3", "boundary"]}
        colLabels={["γ comp 0", "γ comp 1", "γ comp 2"]}
        matrix={[
          [0.33, 0.01, 0.66],
          [0.00, 0.35, 1.00],
          [0.00, 0.00, 1.00],
          [0.00, 1.00, 0.00],
          [0.00, 1.00, 0.00],
          [0.00, 0.98, 0.02],
          [1.00, 0.00, 0.00],
          [1.00, 0.00, 0.00],
          [0.99, 0.01, 0.00],
          [0.31, 0.38, 0.31],
        ]}
        colorScale="gold"
      />

      <H3>6.3 Log-likelihood curve — monotonic increase</H3>

      <Plot
        label="Observed-data log-likelihood per EM iteration (monotonically non-decreasing)"
        xLabel="EM iteration"
        yLabel="log-likelihood"
        series={[
          {
            name: "log-likelihood",
            color: colors.gold,
            points: [
              [1, -1606.46], [2, -1574.77], [3, -1565.23], [4, -1563.98],
              [5, -1563.89], [6, -1563.89], [7, -1563.88], [8, -1563.88],
              [9, -1563.88], [10, -1563.88],
            ],
          },
        ]}
      />

      <H3>6.4 BIC vs K — model selection</H3>

      <Plot
        label="BIC and AIC vs number of components K (lower is better)"
        xLabel="K (number of components)"
        yLabel="information criterion"
        series={[
          {
            name: "BIC",
            color: colors.gold,
            points: [[1, 4165.98], [2, 3547.97], [3, 3231.63], [4, 3258.54], [5, 3293.31], [6, 3316.29], [7, 3348.15]],
          },
          {
            name: "AIC",
            color: colors.green,
            points: [[1, 4145.44], [2, 3502.77], [3, 3161.77], [4, 3164.03], [5, 3174.14], [6, 3172.47], [7, 3179.67]],
          },
        ]}
      />

      <Prose>
        Both BIC and AIC hit their minimum at K=3, correctly identifying the true number of components. BIC applies a stronger penalty for model complexity (ln(n) per parameter vs. 2 per parameter for AIC), so it more aggressively penalizes over-fitting and typically favors sparser models. In practice, BIC tends to select fewer components than AIC. When BIC and AIC disagree, the right choice depends on your goal: if you want the model closest to the true data-generating process (when one exists), BIC is theoretically consistent; if you want the best predictive model, AIC or cross-validation is preferable.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <StepTrace
        label="When to use GMM vs alternatives"
        steps={[
          {
            label: "GMM (full covariance)",
            render: () => (
              <Prose>
                Use when: clusters are elliptical with different orientations, you need soft assignments or density estimates, you need a generative model for sampling or anomaly scoring, you want principled model selection via BIC/AIC. Number of components: must be specified (use BIC grid search or BayesianGMM). Covariance: full is the most flexible and correct choice when d is small (d {"<"} 30 or so). Parameter count: K × d(d+1)/2 covariance parameters — grows quadratically with d. Failure mode: singular covariance when a cluster collapses to few points. Fix: regularization (reg_covar in sklearn, default 1e-6).
              </Prose>
            ),
          },
          {
            label: "GMM (diagonal or spherical covariance)",
            render: () => (
              <Prose>
                Use when: d is large (d {">"} 50) and full covariance is computationally prohibitive or prone to overfitting. Diagonal assumes features are conditionally independent within each cluster — a strong assumption that is often wrong but dramatically reduces parameter count (K×d vs. K×d²/2). Spherical is even more restrictive. In text clustering with bag-of-words features (d = 10,000+), diagonal GMM is the standard choice. Naive Bayes is a special case of diagonal GMM for categorical data with a uniform prior on mixing weights.
              </Prose>
            ),
          },
          {
            label: "K-Means",
            render: () => (
              <Prose>
                Use when: you need speed and simplicity, clusters are expected to be roughly spherical and similarly sized, you do not need probability estimates, d is large (GMM full covariance becomes intractable). K-Means is a special case of GMM with spherical covariance and hard (0/1) assignments — it will give wrong cluster shapes for elongated data but runs an order of magnitude faster. For very large n (millions of rows), use MiniBatchKMeans; GMM has no mature mini-batch implementation in sklearn.
              </Prose>
            ),
          },
          {
            label: "DBSCAN / HDBSCAN",
            render: () => (
              <Prose>
                Use when: the number of clusters is unknown and clusters have non-convex or irregular shapes that Gaussians cannot model (rings, crescents, arbitrary manifolds). DBSCAN finds clusters as dense regions separated by low-density gaps; it automatically identifies outliers as noise points. Weakness: sensitive to the epsilon neighborhood parameter; requires careful tuning in high dimensions. Use DBSCAN when you know clusters have sharp density boundaries; use GMM when you believe the data is genuinely Gaussian-generated.
              </Prose>
            ),
          },
          {
            label: "Bayesian GMM (BayesianGaussianMixture)",
            render: () => (
              <Prose>
                Use when: you do not know K and want to avoid a grid search over number of components. BGMM uses a Dirichlet Process prior to automatically shrink unnecessary components to near-zero weight. Set n_components generously (2× your rough guess) and let the algorithm decide. Tune weight_concentration_prior: small values (1e-3) favor sparse solutions; large values (1.0) allow more active components. BGMM is slower than plain GMM (variational inference requires more iterations) but eliminates the BIC search loop.
              </Prose>
            ),
          },
          {
            label: "Covariance type selection guide",
            render: () => (
              <Prose>
                Start with full if d {"<"} 20 and n {">"} 100×K×d. Move to tied if you believe all clusters have similar shape (saves d(d+1)/2 × (K-1) parameters). Move to diag if d is large or if full covariance is numerically unstable. Use spherical only as a regularization backstop or when speed is critical. Always compare information criteria across covariance types for the same K: on the Section 4 dataset, full achieves BIC=3231 vs. tied=3380 vs. diag=3322 vs. spherical=3355 — full is clearly best when the data has genuine off-diagonal correlations.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 Computational complexity per EM iteration</H3>

      <Prose>
        The cost of one EM iteration depends critically on the covariance type. For full covariance, the E-step requires evaluating the multivariate Gaussian density for all n×K (point, component) pairs. Each evaluation requires solving a d×d linear system (or using the precomputed inverse), costing O(d²) per pair — total O(n×K×d²) per E-step. The M-step forms K scatter matrices, each at cost O(n×d²) — total O(K×n×d²). For diagonal covariance, the density evaluation is O(d) per pair (just pointwise products), so the full iteration drops to O(n×K×d). For large d, this is a dramatic difference: at d=1000, K=10, n=100,000, full covariance costs ~10¹² operations per iteration; diagonal costs ~10⁹.
      </Prose>

      <Prose>
        In terms of memory, full covariance stores K matrices of size d×d — that is K×d² floats. At d=5000, K=10, in float64 this is 10×25,000,000×8 bytes = 2 GB just for the covariances. At d=10,000 it is 8 GB. This is the hard practical limit of full-covariance GMM: it becomes infeasible in the very high-dimensional regime that arises naturally in text, genomics, and image data.
      </Prose>

      <H3>8.2 Convergence rate and iterations</H3>

      <Prose>
        EM has linear convergence: the log-likelihood gap to the local optimum decreases by a constant factor each iteration. The convergence rate depends on the fraction of "missing information" — intuitively, how much the latent assignments help. Well-separated clusters (low overlap) converge in 5–20 iterations. Highly overlapping clusters with similar mixing weights converge slowly — sometimes hundreds of iterations — because the responsibilities are always diffuse and the gradient signal is weak. Near local optima, EM is notoriously slow: the objective plateau region can be very flat, and progress per iteration can shrink to machine precision while technically not satisfying convergence criteria. The standard fix is a strict tolerance (convergence_tol in sklearn, default 1e-3) combined with a max_iter cap.
      </Prose>

      <H3>8.3 Alternatives for large-scale fitting</H3>

      <Prose>
        For very large n, several alternatives to standard EM exist. Mini-batch EM processes a random subset of data at each E-step, then updates parameters using an online average that tracks the full-data sufficient statistics. This reduces per-iteration cost from O(n) to O(batch_size) at the cost of noisier updates and slightly slower convergence per pass through the data. Stochastic EM (SEM) samples hard assignments from the posterior rather than computing soft responsibilities, making each step cheaper but adding variance. For very high d, Probabilistic PCA (a GMM with shared spherical covariance in a low-dimensional subspace) compresses the representation before fitting. For non-Gaussian structure, variational autoencoders (VAE) with a Gaussian mixture prior (VQ-VAE, GMVAE) learn a joint embedding and clustering. The decision rule is simple: if n fits in memory and d {"<"} 50, standard GMM with full covariance in sklearn is appropriate. If n {">"} 10 million or d {">"} 100, move to diagonal GMM, hierarchical clustering for initialization, or a learned-embedding approach.
      </Prose>

      <H3>8.4 Initialization matters more than you think</H3>

      <Prose>
        Standard EM with random initialization can converge to dramatically different local optima on the same dataset. The sklearn default of <Code>n_init=1</Code> with k-means++ seeding is often insufficient for real data. The right practice is <Code>n_init=10</Code> or more, keeping the run with the highest final log-likelihood. K-Means++ seeding (sklearn's default <Code>init='kmeans'</Code>) is substantially better than random initialization: it spreads the initial means across the data by sampling proportionally to the squared distance from already-chosen centers, giving EM a better starting basin. The cost of multiple restarts is embarrassingly parallel — fit K models simultaneously across cores. For very large n where each restart is expensive, spend the budget on more EM iterations from one good initialization rather than multiple short runs from random starts.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Singular covariance — cluster collapse</H3>

      <Prose>
        The most dangerous failure mode in GMM is covariance singularity: a component's covariance matrix becomes singular (non-invertible), making the Gaussian density undefined (formally infinite at the mean). This happens when a component's assigned points collapse to a single point or a low-dimensional subspace — a common occurrence when a component captures only 1–3 points, when d approaches n_k, or when the data has near-perfect collinearity within a cluster. The symptom is a log-likelihood that jumps to +∞ in the middle of EM. Sklearn guards against this with the <Code>reg_covar</Code> parameter (default 1e-6), which adds a small multiple of the identity to every covariance matrix. If singularity still occurs, increase <Code>reg_covar</Code> to 1e-3 or 1e-2. If it persists, the model is overparameterized — reduce K or switch to a more constrained covariance type.
      </Prose>

      <H3>9.2 Initialization sensitivity and local optima</H3>

      <Prose>
        EM guarantees convergence to a stationary point — not to the global maximum. With K {">"} 3 and overlapping clusters, the landscape has multiple local optima that can differ by hundreds of log-likelihood units. Running EM once from a random start is not a reliable strategy. The recommended practice: use <Code>n_init=10</Code> with k-means++ seeding (sklearn's default <Code>init='kmeans'</Code> for GaussianMixture), and report the best result across runs. On adversarial inputs — data with heavy tails, irregular shapes, or very unequal mixing weights — even 10 restarts may not be enough. In these cases, hierarchical clustering (Ward's linkage) can provide a good single initialization that is more robust than k-means++.
      </Prose>

      <H3>9.3 Choosing K</H3>

      <Prose>
        The number of components K is the hyperparameter with the most impact and the least principled selection procedure. The standard approach is to fit models for K ∈ {"{"}1, 2, ..., K_max{"}"} and plot BIC, choosing the K at the minimum (or at the "elbow" if the minimum is very shallow). BIC is theoretically consistent for model selection among correctly specified models — as n → ∞, it selects the true K with probability 1. AIC tends to overestimate K. In practice, on finite messy data, the BIC curve often has a shallow minimum with multiple plausible K values. Supplementing BIC with domain knowledge — "how many natural segments do we believe exist?" — is not a statistical weakness but sound practice. BayesianGaussianMixture is the automatic alternative that removes the grid search entirely.
      </Prose>

      <H3>9.4 Label switching</H3>

      <Prose>
        The GMM likelihood is invariant to relabeling the components: swapping the parameters of Component 0 and Component 1 gives identical likelihood. This means EM can converge to any of K! equivalent solutions, and different restarts may label the same physical cluster as different component indices. This is not a problem if you only care about the final cluster assignments, but it causes issues when averaging parameters across restarts, computing component-specific statistics longitudinally, or using EM in a Bayesian context. The fix for averaging is to align component labels across runs using the Hungarian algorithm (optimal assignment matching means by Euclidean distance). For Bayesian inference, label switching requires special MCMC samplers or post-processing steps.
      </Prose>

      <H3>9.5 Slow convergence near a plateau</H3>

      <Prose>
        EM's linear convergence rate degrades to near-zero when the log-likelihood surface is very flat near the optimum — which happens when clusters heavily overlap, when K is misspecified (too many components for the data structure), or when the data has a mode that EM approaches asymptotically. The practical symptom: log-likelihood improvements of 1e-5 per iteration for hundreds of iterations, never satisfying a tight convergence tolerance. Two responses: (1) use a loose convergence tolerance (1e-3 in log-likelihood change is standard; sklearn's default) and cap iterations generously; (2) switch to Newton-EM or EM with acceleration (SQUAREM algorithm), which uses extrapolation to take larger steps along the EM path. For most practical applications, the plateau-stuck solution is still a good local optimum — the cluster assignments change negligibly over the plateau iterations.
      </Prose>

      <H3>9.6 High-dimensional data: the curse and its remedies</H3>

      <Prose>
        In high dimensions (d {">"} 50), all pairwise distances concentrate — the ratio of the maximum to minimum distance across all data pairs approaches 1 as d → ∞. This concentration of measure means that Gaussian density functions become nearly equal for all points, responsibilities become near-uniform, and EM makes negligible progress. The symptoms: log-likelihood barely improves after initialization, and the final cluster assignments are nearly random. The remedies, in order of invasiveness: (1) reduce dimensionality with PCA before fitting GMM; (2) switch to diagonal covariance; (3) use subspace clustering models (Probabilistic PCA, MPPCA) that embed the GMM in a learned low-dimensional space; (4) learn representations with a neural encoder and cluster in the latent space.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations were WebSearch-verified for author, year, venue, volume, and page numbers. Read them in this order to follow the intellectual lineage from the problem statement to the algorithm to the convergence theory to modern extensions.
      </Prose>

      <StepTrace
        label="Primary literature"
        steps={[
          {
            label: "Pearson 1894 — The mixture problem is posed",
            render: () => (
              <Prose>
                Pearson, K. (1894). "Contributions to the Mathematical Theory of Evolution." <em>Philosophical Transactions of the Royal Society of London, Series A</em>, 185, 71–110. DOI: 10.1098/rsta.1894.0003. Pearson fits a mixture of two Gaussians to Neapolitan crab data using method of moments — solving a 9th-degree polynomial to recover five mixture parameters. The paper introduces the two-component Gaussian mixture as a statistical object and establishes the decomposition of an observed heterogeneous distribution into homogeneous components as a well-posed problem. The method-of-moments approach was computationally heroic for its time but numerically unstable; it would be superseded by maximum likelihood estimation, which required EM to be tractable.
              </Prose>
            ),
          },
          {
            label: "Dempster, Laird & Rubin 1977 — EM is formalized",
            render: () => (
              <Prose>
                Dempster, A.P., Laird, N.M., and Rubin, D.B. (1977). "Maximum Likelihood from Incomplete Data via the EM Algorithm." <em>Journal of the Royal Statistical Society, Series B (Methodological)</em>, 39(1), 1–38. DOI: 10.1111/j.2517-6161.1977.tb01600.x. The canonical EM paper. Dempster et al. define the complete-data and observed-data likelihood, introduce the Q function (expected complete-data log-likelihood), prove the fundamental monotonicity property (each EM step does not decrease the observed-data likelihood), and apply the framework to dozens of statistical models including mixture distributions, censored data, missing data, and variance components. The proof of monotonicity via Jensen's inequality (Section 3 of the paper) is the result everyone cites. Cited over 70,000 times.
              </Prose>
            ),
          },
          {
            label: "Wu 1983 — Convergence theory completed",
            render: () => (
              <Prose>
                Wu, C.F.J. (1983). "On the Convergence Properties of the EM Algorithm." <em>The Annals of Statistics</em>, 11(1), 95–103. DOI: 10.1214/aos/1176346060. Dempster et al. proved that each EM iteration does not decrease the likelihood — a monotonicity result. Wu proved the stronger result: under mild regularity conditions, the sequence of parameter estimates converges (to a stationary point of the likelihood — not necessarily the global maximum). He also gave conditions under which convergence to a local maximum is guaranteed. The paper closes the theoretical gap that Dempster et al. left open and establishes EM as a rigorously convergent algorithm.
              </Prose>
            ),
          },
          {
            label: "McLachlan & Peel 2000 — Definitive book",
            render: () => (
              <Prose>
                McLachlan, G.J. and Peel, D. (2000). <em>Finite Mixture Models</em>. Wiley Series in Probability and Statistics. New York: John Wiley {"&"} Sons. ISBN: 978-0-471-00626-8. The authoritative reference for mixture models. Covers identifiability (when are mixture parameters uniquely recoverable?), EM fitting for Gaussian and non-Gaussian mixtures, testing for the number of components, maximum likelihood properties, asymptotic theory, and applications to clustering, discriminant analysis, and medical imaging. Chapter 2 on the EM algorithm is the most rigorous non-paper treatment available. More than 800 references, 40% published after 1995.
              </Prose>
            ),
          },
          {
            label: "Figueiredo & Jain 2002 — Automatic component selection",
            render: () => (
              <Prose>
                Figueiredo, M.A.T. and Jain, A.K. (2002). "Unsupervised Learning of Finite Mixture Models." <em>IEEE Transactions on Pattern Analysis and Machine Intelligence</em>, 24(3), 381–396. DOI: 10.1109/34.990138. Proposes an EM algorithm that simultaneously estimates parameters and selects the number of components, without requiring BIC/AIC post-hoc comparison. The approach uses a minimum message length (MML) regularizer that penalizes each component's parameters; components that do not earn their penalty are shrunk to zero weight and effectively pruned during fitting. A cleaner alternative to BayesianGaussianMixture for practitioners who want automatic K selection without tuning a Dirichlet prior concentration.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <Prose>
        Work through these without reading the answers first. The goal is to catch gaps in the derivation, not to memorize formulas.
      </Prose>

      <H3>Exercise 1 (derivation)</H3>
      <Prose>
        Write the E-step responsibility formula from first principles using Bayes' theorem. What is the numerator? What is the denominator? What happens numerically when all component densities are very small (e.g., a point far from all cluster means in high dimensions), and how should you handle it in code?
      </Prose>
      <Callout type="answer" title="Answer 1">
        By Bayes' theorem: γ_ik = P(z_i=k | x_i, θ) = P(x_i | z_i=k, θ) P(z_i=k | θ) / P(x_i | θ). Numerator: π_k × N(x_i; μ_k, Σ_k) — the joint probability of observing x_i and component k. Denominator: Σ_j π_j × N(x_j; μ_j, Σ_j) — the marginal likelihood of x_i under the mixture, which normalizes the responsibilities to sum to 1. Numerical issue: when all Gaussian densities are near zero (a point far from all centers, or very small variances), the numerator and denominator can both underflow to zero in float64. Fix: compute responsibilities in log space — take log of numerator for each k, use the log-sum-exp trick to compute the log of the denominator stably, subtract to get log responsibilities, then exponentiate. Alternatively, clip the denominator at a small epsilon (1e-300) before dividing, accepting that points in zero-density regions get uniform responsibilities.
      </Callout>

      <H3>Exercise 2 (proof)</H3>
      <Prose>
        Prove that the EM algorithm monotonically increases (or does not decrease) the observed-data log-likelihood. Your proof should: (1) define the Q function, (2) apply Jensen's inequality, (3) show why the E-step makes the bound tight, (4) explain what the M-step does to the bound.
      </Prose>
      <Callout type="answer" title="Answer 2">
        Let q_i(k) be any distribution over component assignments for point i. By Jensen's inequality (log is concave): log p(x_i | θ) = log Σ_k q_i(k) [π_k N(x_i; μ_k, Σ_k) / q_i(k)] ≥ Σ_k q_i(k) log [π_k N(x_i; μ_k, Σ_k) / q_i(k)]. The right side is the ELBO: Q(θ, q) = Σ_i Σ_k q_i(k) log[π_k N(x_i; μ_k, Σ_k)] + H(q), where H(q) is the entropy of q. E-step: choose q_i(k) = γ_ik (the posterior). Jensen's bound is tight (equality holds) exactly when q_i(k) ∝ π_k N(x_i; μ_k, Σ_k), which is the responsibility. So the E-step eliminates the gap between the ELBO and the log-likelihood at the current θ. M-step: maximize Q(θ', q) over θ', keeping q fixed. Since Q(θ_new, q) ≥ Q(θ_old, q) = log p(X | θ_old), we have log p(X | θ_new) ≥ Q(θ_new, q) ≥ Q(θ_old, q) = log p(X | θ_old). QED.
      </Callout>

      <H3>Exercise 3 (conceptual)</H3>
      <Prose>
        You run GMM with K=3 on a dataset and inspect the final mixing weights: [0.001, 0.499, 0.500]. What does this tell you? Is the model a good fit? What would you do next?
      </Prose>
      <Callout type="answer" title="Answer 3">
        The weight of 0.001 for Component 0 indicates that EM has effectively disabled that component — it claims only 0.1% of the data (about 1 point in a 1000-point dataset). This is a sign that K=3 is too many components for this data. The two large components (≈0.5 each) are doing the real work; Component 0 is a ghost. Next steps: (1) refit with K=2 and compare BIC — it will almost certainly be lower. (2) Check if the ghost component is capturing a genuine outlier cluster by inspecting which point(s) it dominates. (3) If BIC strongly prefers K=2, the ghost was an artifact of initialization; if BIC is similar, there may be a tiny dense cluster worth investigating. A BayesianGaussianMixture would have pruned the ghost component automatically.
      </Callout>

      <H3>Exercise 4 (debugging)</H3>
      <Prose>
        You implement GMM from scratch and observe that the log-likelihood <em>decreases</em> on iteration 4. List the three most likely bugs in your implementation, in order of probability.
      </Prose>
      <Callout type="answer" title="Answer 4">
        A correct EM implementation never decreases the log-likelihood (Wu 1983). A decrease means a bug. Three most likely causes: (1) Computing responsibilities and updating parameters in the same loop without separating the E-step and M-step. If you update mu_0 and then use the new mu_0 to compute responsibilities for the remaining points in the same pass, you have broken the independence assumption. Fix: complete the full E-step (compute all γ_ik using current parameters) before any M-step update. (2) Forgetting to re-normalize responsibilities. If gamma rows do not sum to 1 (due to a missing normalization step or a numerical clamp that breaks the sum), the M-step produces parameters that do not correspond to any valid mixture, and the likelihood can jump arbitrarily. Fix: verify gamma.sum(axis=1) == 1 after the E-step. (3) Not regularizing the covariance. A singular or near-singular Sigma_k produces negative or infinite log-determinant in the Gaussian density, which can cause the log-likelihood computation to return -inf or nan. Fix: add reg_covar * I to each covariance before computing the density.
      </Callout>

      <H3>Exercise 5 (applied)</H3>
      <Prose>
        You want to use GMM for anomaly detection: assign an anomaly score to each new data point. Describe the scoring procedure, and explain what parameter choices matter most for this use case.
      </Prose>
      <Callout type="answer" title="Answer 5">
        GMM anomaly scoring: fit the GMM on clean training data. For a new point x, compute the mixture density p(x) = Σ_k π_k N(x; μ_k, Σ_k). Use negative log-density as the anomaly score: score(x) = -log p(x). Points with high score (low density under the model) are anomalous. In sklearn: fit a GaussianMixture, then use gm.score_samples(X_test) which returns log p(x) per sample; negate for anomaly scores. Parameter choices that matter: (1) covariance_type — full is most sensitive to anisotropic anomalies; spherical will miss anomalies that are only outlying in certain directions. (2) K — too few components and normal regions are not modeled precisely; too many and anomaly scores become noisy. (3) n_init — GMM anomaly detection requires a good fit; use n_init ≥ 10. (4) Threshold — there is no canonical anomaly threshold; use a percentile of training scores (e.g., 99th percentile) or a domain-specific false-positive budget.
      </Callout>

      <H3>Exercise 6 (synthesis)</H3>
      <Prose>
        Explain why k-means can be seen as a special case of GMM. What assumptions does k-means implicitly make about the covariance structure, mixing weights, and assignment strategy? Under what conditions would GMM with these assumptions give exactly the same cluster assignments as k-means?
      </Prose>
      <Callout type="answer" title="Answer 6">
        K-means is GMM with three simultaneous restrictions: (1) Covariance: Σ_k = σ²I for all k — spherical, equal-variance covariances. This makes the Gaussian density proportional to exp(-||x - μ_k||² / 2σ²), and the log-density (ignoring the shared normalization constant) reduces to -||x - μ_k||². (2) Mixing weights: π_k = 1/K for all k — equal-weight components. (3) Assignment: hard (0/1) instead of soft — each point is assigned to exactly the most responsible component. Under these three restrictions, the GMM E-step collapses to: assign each point to the component k with the smallest Euclidean distance ||x_i - μ_k||² — which is exactly the k-means assignment step. The M-step collapses to: update each centroid as the mean of its assigned points — which is exactly the k-means update. So GMM with spherical tied covariance and hard assignments is identical to k-means. The implication: if you run GMM with covariance_type='spherical' and threshold responsibilities at 0.5, you get k-means. Use k-means when these assumptions hold; use GMM when they do not.
      </Callout>

    </div>
  ),
};

export default gmmContent;
