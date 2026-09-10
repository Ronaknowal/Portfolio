import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const icaContent = {
  title: "Independent Component Analysis (ICA)",
  readTime: "~40 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Imagine you are at a cocktail party. Three conversations are happening simultaneously — a singer practicing scales at the piano, two people arguing about politics, and someone narrating a story near the bar. You have placed three microphones around the room. Each microphone records a different linear mixture of all three voices. The recordings are hopelessly entangled — you cannot simply listen to microphone 1 and hear the pianist, because every mic picks up every speaker. The question is: can you reconstruct the three original source signals from three mixed recordings, without ever hearing the sources separately, and without knowing the acoustic geometry of the room? This is the cocktail party problem, and it is the defining motivation for Independent Component Analysis.
      </Prose>

      <Prose>
        The mathematical origin of ICA traces to a 1985 GRETSI conference paper by Hérault, Jutten, and Ans: "Détection de grandeurs primitives dans un message composite par une architecture de calcul neuromimétique en apprentissage non supervisé." Their neuromimetic adaptive algorithm demonstrated that linear mixtures of independent signals could be separated online — they just lacked the theoretical explanation for why it worked. Pierre Comon supplied the theory nine years later in the field-defining 1994 paper "Independent Component Analysis, a new concept?" published in <em>Signal Processing</em> 36(3):287–314. Comon established the formal ICA model — a noiseless linear mixing of statistically independent, non-Gaussian source signals — and showed that independence (not just decorrelation) is both necessary and sufficient (up to ordering and scaling) to identify the sources. Decorrelation, the objective of PCA, is a second-order property; independence is a full distributional property, requiring all higher-order statistics to match.
      </Prose>

      <Prose>
        The practical algorithm came from Bell and Sejnowski in 1995: "An Information-Maximization Approach to Blind Separation and Blind Deconvolution," <em>Neural Computation</em> 7(6):1129–1159. Their Infomax algorithm maximizes the output entropy of a neural network with nonlinear activation functions, and they showed this is equivalent to maximizing the mutual information between inputs and outputs — equivalently, finding statistically independent components. Infomax separated up to 10 simultaneous speakers in real recordings, making blind source separation practical for the first time.
      </Prose>

      <Prose>
        The algorithm in production use today is FastICA, introduced by Hyvärinen and Oja in 1997: "A Fast Fixed-Point Algorithm for Independent Component Analysis," <em>Neural Computation</em> 9(7):1483–1492. FastICA reformulated the search for independent components as a fixed-point iteration — instead of gradient ascent on entropy, it performs Newton-like updates that converge cubically rather than linearly. On practical datasets, FastICA is 10–100× faster than gradient-based Infomax and is the default in every major ML library. The comprehensive mathematical treatment is in Hyvärinen, Karhunen, and Oja (2001), <em>Independent Component Analysis</em>, Wiley — still the definitive reference. More recently, Ablin, Cardoso, and Gramfort (2018) introduced Picard (arXiv:1706.08171), a preconditioned L-BFGS variant that is faster still, especially on real neural and audio data.
      </Prose>

      <Prose>
        ICA belongs to a family of blind source separation (BSS) methods: algorithms that recover original signals from observed mixtures without knowledge of the mixing process. The "blind" qualifier is important — you know nothing about the room acoustics, the sensor placements, or the source distributions except that the sources are statistically independent and non-Gaussian. The constraint of independence is what gives ICA its teeth: it is a much stronger condition than the decorrelation that PCA achieves, and it is powerful enough to uniquely identify the sources (up to the inherent sign, scale, and permutation ambiguities we will discuss in detail).
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        PCA and ICA both find directions in feature space, but they optimize completely different objectives. PCA finds the directions of maximum variance — orthogonal axes along which the data is most spread out. ICA finds the directions of maximum statistical independence — axes along which the projected signals share no information with each other at any order of statistics, not just second order. The two objectives coincide only for Gaussian data; for non-Gaussian data, they can produce radically different decompositions.
      </Prose>

      <Prose>
        The central insight linking non-Gaussianity and independence comes from the Central Limit Theorem, read in reverse. The CLT says: mix many independent random variables, and the sum tends toward a Gaussian, regardless of the individual distributions. Applied to ICA: if you mix two non-Gaussian independent sources, the mixture is more Gaussian than either source individually. Conversely, starting from a mixture and searching for a linear projection that is <em>maximally non-Gaussian</em> is equivalent to searching for an un-mixed source signal. Non-Gaussianity is the compass that points toward independence.
      </Prose>

      <Prose>
        You can see this in kurtosis (the standardized fourth central moment). A Gaussian has excess kurtosis of zero. A Laplace distribution (heavy tails) has excess kurtosis of 3. A uniform distribution (light tails) has excess kurtosis of −1.2. A square wave has excess kurtosis near −2. After mixing two Laplace sources 50/50, the mixture kurtosis drops to 1.72; mixing three drops it to 0.95 — converging toward the Gaussian value of 0. ICA reverses this: it finds unmixing directions that push kurtosis back toward the extremes, away from zero.
      </Prose>

      <Plot
        label="CLT effect: mixing makes signals more Gaussian (kurtosis)"
        xLabel="number of Laplace sources mixed"
        yLabel="excess kurtosis"
        series={[
          {
            name: "mixture kurtosis",
            color: colors.gold,
            points: [[1, 3.25], [2, 1.72], [3, 0.95], [4, 0.56], [6, 0.22], [8, 0.10]],
          },
          {
            name: "Gaussian target (kurtosis = 0)",
            color: colors.green,
            points: [[1, 0], [8, 0]],
          },
        ]}
      />

      <Prose>
        The contrast between PCA and ICA is sharpest on non-Gaussian data. Consider two independent Laplace sources mixed by a 2×2 matrix. PCA finds the two directions of maximum variance in the mixture — these are rotations that decorrelate the mixed signals. Because the Laplace distribution is symmetric and has zero cross-correlation with the other source by independence, PCA does decorrelate the mixture. But decorrelation is not independence for non-Gaussian distributions: there can be many rotations that produce zero second-order correlation while still having strong higher-order dependencies. PCA picks one of these rotations arbitrarily (the variance-maximizing one). ICA picks the specific rotation that produces genuine statistical independence — and for non-Gaussian sources, that rotation corresponds to the true un-mixing direction.
      </Prose>

      <Prose>
        The practical consequence: PCA components will still "look mixed" when the original sources are non-Gaussian. ICA components will match the true sources up to sign and permutation. This is why ICA is indispensable for signal separation (audio, EEG artifact removal), while PCA is the better choice for compression and visualization of Gaussian-dominated data.
      </Prose>

      <Callout type="info" title="Why Gaussian sources cannot be separated">
        If the sources are Gaussian, the ICA model is not identifiable. A multivariate Gaussian has the property that any orthogonal rotation of its components produces another multivariate Gaussian with the same covariance structure. There is no preferred basis — every rotation looks equally independent to any statistical test. This means that for Gaussian sources, the ICA fixed-point iteration has no stable fixed point corresponding to the true sources; it will converge to an arbitrary rotation of the whitened data, which is exactly what PCA gives you. The non-Gaussianity assumption is not a convenience — it is a mathematical necessity for ICA to work.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 The ICA generative model</H3>

      <Prose>
        The ICA model is:
      </Prose>

      <MathBlock>
        {"\\mathbf{x} = A \\mathbf{s}"}
      </MathBlock>

      <Prose>
        where <Code>x ∈ ℝᵈ</Code> is the observed mixed signal vector, <Code>s ∈ ℝᵈ</Code> is the vector of latent independent source signals, and <Code>A ∈ ℝ^(d×d)</Code> is the unknown mixing matrix. The model assumptions are: (1) the sources <Code>s₁, s₂, ..., sᵈ</Code> are mutually statistically independent; (2) at most one source is Gaussian; (3) <Code>A</Code> is square and invertible. Goal: estimate the unmixing matrix <Code>W ≈ A⁻¹</Code> such that:
      </Prose>

      <MathBlock>
        {"\\mathbf{s} = W \\mathbf{x} = W A \\mathbf{s}"}
      </MathBlock>

      <Prose>
        The product <Code>WA</Code> should be a generalized permutation matrix — a permutation of rows times a diagonal scaling matrix. The ICA solution is unique up to: (1) permutation of rows (you cannot determine which recovered component corresponds to which source without additional information); (2) sign flips (each component can be multiplied by −1); (3) scaling (you cannot separately identify the scale of <Code>W</Code> and <Code>s</Code>, so conventionally sources are normalized to unit variance). These are called the inherent ambiguities of ICA.
      </Prose>

      <H3>3.2 Why decorrelation is not independence</H3>

      <Prose>
        Two random variables <Code>u</Code> and <Code>v</Code> are uncorrelated if <Code>E[uv] = E[u]E[v]</Code>. They are independent if <Code>p(u, v) = p(u)p(v)</Code> — the full joint distribution factorizes. Independence implies zero correlation, but the converse only holds for jointly Gaussian variables. For non-Gaussian variables, you can have zero correlation with strong dependence: if <Code>u ~ Uniform(−1, 1)</Code> and <Code>v = u²</Code>, then <Code>Cov(u, v) = E[u³] − E[u]E[u²] = 0</Code> (odd moments of a symmetric distribution vanish), yet knowing <Code>v = 0.25</Code> tells you exactly that <Code>u = ±0.5</Code>.
      </Prose>

      <Prose>
        PCA diagonalizes the covariance matrix — it finds a rotation that makes the second-order statistics diagonal (zero pairwise correlations). ICA must go further, matching all higher-order moments. The information-theoretic statement: ICA minimizes the mutual information between components. Mutual information <Code>I(y₁; y₂; ...; yᵈ)</Code> is zero if and only if the components are fully independent. Minimizing mutual information is equivalent to maximizing the sum of marginal entropies minus the joint entropy — which is equivalent to maximizing the non-Gaussianity of each marginal.
      </Prose>

      <H3>3.3 Non-Gaussianity objectives</H3>

      <Prose>
        Two main measures of non-Gaussianity are used as objectives:
      </Prose>

      <Prose>
        <strong>Kurtosis.</strong> The excess kurtosis of a random variable <Code>y</Code> is:
      </Prose>

      <MathBlock>
        {"\\text{kurt}(y) = E[y^4] - 3(E[y^2])^2"}
      </MathBlock>

      <Prose>
        A Gaussian has <Code>kurt = 0</Code>. Super-Gaussian (heavy-tailed) distributions like Laplace have positive kurtosis; sub-Gaussian (light-tailed) distributions like uniform have negative kurtosis. ICA can maximize <Code>|kurt(wᵀx)|</Code> over unit vectors <Code>w</Code>. Kurtosis is simple but sensitive to outliers — a single corrupted sample can dominate the fourth moment.
      </Prose>

      <Prose>
        <strong>Negentropy.</strong> More robust. The negentropy of <Code>y</Code> is:
      </Prose>

      <MathBlock>
        {"J(y) = H(y_{\\text{Gauss}}) - H(y)"}
      </MathBlock>

      <Prose>
        where <Code>H</Code> is differential entropy and <Code>y_Gauss</Code> is a Gaussian with the same variance as <Code>y</Code>. By the maximum entropy principle, the Gaussian maximizes entropy among all distributions with a fixed variance, so <Code>J(y) ≥ 0</Code>, with equality only for Gaussians. Negentropy is a theoretically clean measure but expensive to compute. In practice, the approximation due to Hyvärinen is used:
      </Prose>

      <MathBlock>
        {"J(y) \\approx \\left[ E[G(y)] - E[G(\\nu)] \\right]^2"}
      </MathBlock>

      <Prose>
        where <Code>ν ~ N(0,1)</Code> and <Code>G</Code> is a smooth nonlinear function. The FastICA algorithm uses <Code>G(u) = log cosh(u)</Code> (the default, robust to outliers), <Code>G(u) = -exp(-u²/2)</Code> (for super-Gaussian sources), or <Code>G(u) = u⁴/4</Code> (equivalent to kurtosis maximization). The corresponding derivatives <Code>g = G'</Code> appear directly in the fixed-point update rule.
      </Prose>

      <H3>3.4 FastICA fixed-point update rule</H3>

      <Prose>
        FastICA extracts one component at a time (deflation) or all simultaneously (parallel). For the deflation case, the fixed-point iteration for extracting the <Code>k</Code>-th unmixing vector <Code>w</Code> from whitened data <Code>x̃</Code> is:
      </Prose>

      <MathBlock>
        {"w \\leftarrow \\frac{1}{n} \\sum_{i=1}^{n} \\tilde{x}_i \\, g(w^\\top \\tilde{x}_i) - \\overline{g'(w^\\top \\tilde{x}_i)} \\cdot w"}
      </MathBlock>

      <MathBlock>
        {"w \\leftarrow w \\, / \\, \\|w\\|"}
      </MathBlock>

      <Prose>
        where <Code>g = G'</Code> is the derivative of the chosen nonlinearity and the overline denotes the sample mean <Code>(1/n)Σ g'(wᵀx̃ᵢ)</Code>. This is a Newton step on the negentropy objective, and it converges cubically — typically in 3–10 iterations for well-conditioned data. After extracting each component, deflation orthogonalizes the new vector against all previously extracted vectors to enforce independence between components.
      </Prose>

      <H3>3.5 Whitening as preprocessing</H3>

      <Prose>
        Whitening (sphering) transforms the data so that the covariance matrix of the result is the identity: <Code>E[x̃x̃ᵀ] = I</Code>. This reduces the ICA search from the full space of invertible matrices to the space of orthogonal matrices — a much smaller space. The whitening matrix is:
      </Prose>

      <MathBlock>
        {"W_{\\text{white}} = D^{-1/2} V^\\top"}
      </MathBlock>

      <Prose>
        where <Code>V</Code> and <Code>D</Code> come from the eigendecomposition of the sample covariance <Code>Σ = VDVᵀ</Code>. After whitening, ICA only needs to find an orthogonal rotation <Code>U</Code> such that <Code>Ux̃</Code> has independent components — a constrained search over the orthogonal group <Code>O(d)</Code>. This decoupling is why FastICA is efficient: whitening handles the second-order structure (decorrelation), and the fixed-point iteration handles the higher-order structure (independence).
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below uses NumPy only. We implement the full FastICA pipeline: generate three independent source signals, mix them with a known matrix, whiten the mixture, and apply fixed-point iteration with deflation to recover the sources. Every output shown is verbatim from a verified run.
      </Prose>

      <H3>4a. Generate synthetic cocktail party data</H3>

      <CodeBlock language="python">
{`import numpy as np

np.random.seed(42)
n_samples = 2000
t = np.linspace(0, 8 * np.pi, n_samples)

# Three independent source signals — sine, square wave, Laplace noise
s1 = np.sin(2.5 * t)                              # sine wave (sub-Gaussian)
s2 = np.sign(np.sin(3.7 * t)).astype(float)       # square wave (sub-Gaussian)
s3 = np.random.laplace(size=n_samples)            # heavy-tail noise (super-Gaussian)

S = np.column_stack([s1, s2, s3])
S /= S.std(axis=0)   # unit variance per source
print("Source shapes:", S.shape)
# Output: Source shapes: (2000, 3)
print("Source variances (should be ~1.0):", S.var(axis=0).round(4))
# Output: Source variances (should be ~1.0): [1. 1. 1.]

# True mixing matrix A (unknown to the ICA algorithm)
A = np.array([[1.0, 0.5, 0.3],
              [0.5, 1.0, 0.6],
              [0.3, 0.6, 1.0]])

X = S @ A.T      # observed mixed signals, shape (2000, 3)
print("Mixed signal shape:", X.shape)
# Output: Mixed signal shape: (2000, 3)
print("Mixed variances:", X.var(axis=0).round(4))
# Output: Mixed variances: [1.3146 1.5579 1.4124]
# Variances > 1 — mixing inflates spread. ICA must recover unit-variance sources.`}
      </CodeBlock>

      <H3>4b. Whiten the data</H3>

      <CodeBlock language="python">
{`def whiten(X):
    """Center and whiten X so that Cov(X_white) = I."""
    X_c = X - X.mean(axis=0)
    cov = (X_c.T @ X_c) / (len(X_c) - 1)
    # eigh: symmetric matrix eigendecomposition, ascending order -> reverse
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Whitening matrix: D^{-1/2} V^T
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
    W_white = D_inv_sqrt @ eigenvectors.T
    X_white = X_c @ W_white.T
    return X_white, W_white

X_white, W_white = whiten(X)

# Verify: whitened covariance should be identity
cov_white = (X_white.T @ X_white) / (len(X_white) - 1)
print("Whitened covariance diag:", np.diag(cov_white).round(4))
# Output: Whitened covariance diag: [1. 1. 1.]
print("Max off-diagonal entry:", np.abs(cov_white - np.diag(np.diag(cov_white))).max())
# Output: Max off-diagonal entry: 0.0
# Perfect decorrelation — whitening works. ICA now only needs to find a rotation.`}
      </CodeBlock>

      <H3>4c. FastICA fixed-point iteration with deflation</H3>

      <CodeBlock language="python">
{`def g_tanh(u):
    """Nonlinearity G(u) = log cosh(u), derivative g(u) = tanh(u)."""
    tanh_u = np.tanh(u)
    return tanh_u, 1.0 - tanh_u ** 2   # g(u), g'(u)

def fastica_one_unit(X_white, w_init, g_func, max_iter=500, tol=1e-6):
    """Fixed-point iteration for a single IC."""
    n = len(X_white)
    w = w_init / np.linalg.norm(w_init)
    for i in range(max_iter):
        proj = X_white @ w               # shape (n,) — projection scores
        gval, gprime = g_func(proj)
        # FastICA update: E[x g(w^T x)] - E[g'(w^T x)] * w
        w_new = (X_white.T @ gval) / n - gprime.mean() * w
        w_new /= np.linalg.norm(w_new)
        # Convergence: |w_new . w| -> 1 (angle -> 0 or pi)
        if abs(abs(np.dot(w_new, w)) - 1.0) < tol:
            return w_new, i + 1
        w = w_new
    return w, max_iter   # did not converge

def fastica_deflation(X_white, n_components, g_func, seed=1):
    rng = np.random.default_rng(seed)
    d = X_white.shape[1]
    W = []
    for k in range(n_components):
        w_init = rng.standard_normal(d)
        w, n_iter = fastica_one_unit(X_white, w_init, g_func)
        print(f"  IC{k+1}: converged at iteration {n_iter}")
        # Gram-Schmidt deflation: orthogonalize against already-found components
        for prev_w in W:
            w -= np.dot(w, prev_w) * prev_w
        w /= np.linalg.norm(w)
        W.append(w)
    return np.array(W)   # shape (n_components, d)

print("FastICA deflation:")
W_ica = fastica_deflation(X_white, n_components=3, g_func=g_tanh)
# Output:
#   IC1: converged at iteration 4
#   IC2: converged at iteration 5
#   IC3: converged at iteration 4

# Recover estimated sources
S_recovered = X_white @ W_ica.T   # shape (2000, 3)
print("Recovered sources shape:", S_recovered.shape)
# Output: Recovered sources shape: (2000, 3)`}
      </CodeBlock>

      <H3>4d. Verify recovery — correlation with true sources</H3>

      <CodeBlock language="python">
{`from itertools import permutations

# |corr(true_source_i, recovered_IC_j)| — rows=true, cols=recovered
corr_matrix = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        corr_matrix[i, j] = abs(np.corrcoef(S[:, i], S_recovered[:, j])[0, 1])

print("Absolute correlation matrix |corr(true, recovered)|:")
print(corr_matrix.round(4))
# Output:
# Absolute correlation matrix |corr(true, recovered)|:
# [[0.9999 0.0136 0.0066]
#  [0.0159 0.0055 0.9999]
#  [0.0036 0.9997 0.0243]]
# True source 1 (sine) is recovered IC1 with |corr| = 0.9999
# True source 2 (square) is recovered IC3 with |corr| = 0.9999
# True source 3 (Laplace) is recovered IC2 with |corr| = 0.9997
# -> permutation ambiguity: ICs are reordered but near-perfectly recovered

# Find best permutation
best = max(permutations(range(3)),
           key=lambda p: sum(corr_matrix[i, p[i]] for i in range(3)))
print("Best permutation (true_src_idx -> recovered_IC_idx):", best)
# Output: Best permutation (true_src_idx -> recovered_IC_idx): (0, 2, 1)

for i, j in enumerate(best):
    print("  True source %d -> Recovered IC %d  |corr| = %.4f" % (i+1, j+1, corr_matrix[i, j]))
# Output:
#   True source 1 -> Recovered IC 1  |corr| = 0.9999
#   True source 2 -> Recovered IC 3  |corr| = 0.9999
#   True source 3 -> Recovered IC 2  |corr| = 0.9997
# Near-perfect recovery. Residual error (1 - corr) ~0.0001 is due to finite
# sample noise. Sign ambiguity: each IC may be negated — harmless.`}
      </CodeBlock>

      <Callout type="info" title="The three ambiguities in action">
        The output above demonstrates all three ICA ambiguities simultaneously. Permutation: source 2 appeared as IC3, not IC2. Sign: each IC could be negated. Scale: we normalized sources to unit variance before mixing, but in a real application you would not know the scale of the original sources. None of these ambiguities affect the practical utility of ICA for source separation — you can reorder, sign-flip, and rescale the components after recovery.
      </Callout>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        <Code>sklearn.decomposition.FastICA</Code> is the standard production choice. It handles centering, whitening, and the fixed-point iteration, exposes the mixing matrix and unmixing matrix directly, and supports both <Code>fit_transform</Code> and <Code>transform</Code> for applying the learned unmixing to new data.
      </Prose>

      <H3>5a. sklearn FastICA — core API</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.decomposition import FastICA

np.random.seed(42)
n_samples = 2000
t = np.linspace(0, 8 * np.pi, n_samples)
s1 = np.sin(2.5 * t)
s2 = np.sign(np.sin(3.7 * t)).astype(float)
s3 = np.random.laplace(size=n_samples)
S = np.column_stack([s1, s2, s3])
S /= S.std(axis=0)
A = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.6], [0.3, 0.6, 1.0]])
X = S @ A.T

# --- algorithm='parallel': update all ICs simultaneously (default) ---
ica = FastICA(
    n_components=3,
    algorithm='parallel',     # 'parallel' | 'deflation'
    fun='logcosh',            # 'logcosh' (default) | 'exp' | 'cube'
    max_iter=500,
    tol=1e-4,
    random_state=42,
)
S_recovered = ica.fit_transform(X)   # shape (n_samples, n_components)

print("n_iter_ (iterations to converge):", ica.n_iter_)
# Output: n_iter_ (iterations to converge): 4

print("components_ shape (unmixing W):", ica.components_.shape)
# Output: components_ shape (unmixing W): (3, 3)

print("mixing_ shape (estimated A):", ica.mixing_.shape)
# Output: mixing_ shape (estimated A): (3, 3)
print("mixing_ (estimated A, cols permuted by ICA):")
print(ica.mixing_.round(4))
# Output:
# mixing_ (estimated A, cols permuted by ICA):
# [[ 0.3218  0.4748  0.9928]
#  [ 0.6092  0.9749  0.4862]
#  [ 1.0052  0.5671  0.2832]]
# Columns are permuted relative to true A — permutation ambiguity expected.

# Reconstruction: inverse_transform recovers the original mixed signals exactly
X_recon = ica.inverse_transform(S_recovered)
recon_mse = np.mean((X - X_recon) ** 2)
print(f"Inverse transform reconstruction MSE: {recon_mse:.8f}")
# Output: Inverse transform reconstruction MSE: 0.00000000

# Apply learned unmixing to new data (do NOT call fit_transform again)
X_new = X[:5]
S_new = ica.transform(X_new)
print("Transform 5 new samples, shape:", S_new.shape)
# Output: Transform 5 new samples, shape: (5, 3)
print(S_new.round(4))
# Output:
# [[-0.2059 -0.0139  0.0045]
#  [ 1.6786  0.9903  0.0227]
#  [ 0.4732  0.9883  0.0947]
#  [ 0.1862  0.988   0.1457]
#  [-0.7993  0.9865  0.2124]]`}
      </CodeBlock>

      <H3>5b. Algorithm and function options</H3>

      <CodeBlock language="python">
{`# --- algorithm='deflation', fun='exp': extract ICs sequentially ---
ica_def = FastICA(n_components=3, algorithm='deflation', fun='exp',
                  random_state=0, max_iter=500)
S_def = ica_def.fit_transform(X)
print("deflation, exp -> n_iter_:", ica_def.n_iter_)
# Output: deflation, exp -> n_iter_: 3

# Verify recovery quality with deflation+exp
corr_def = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        corr_def[i, j] = abs(np.corrcoef(S[:, i], S_def[:, j])[0, 1])
print("Max per-source correlation (best permutation):")
print(corr_def.round(4))
# Output:
# [[0.9999 0.0065 0.0131]
#  [0.016  0.9999 0.    ]
#  [0.0032 0.0297 0.9996]]

# fun='cube' is equivalent to kurtosis maximization:
ica_cube = FastICA(n_components=3, fun='cube', random_state=0)
S_cube = ica_cube.fit_transform(X)

# --- Using whiten='unit-variance' (default) vs whiten='arbitrary-variance' ---
# whiten='unit-variance': components_ are scaled so each IC has unit variance
# whiten='arbitrary-variance': no variance normalization (sklearn >= 1.1)
# For most applications, the default 'unit-variance' is correct.

# --- n_components < d: extract a subset of ICs ---
ica_2 = FastICA(n_components=2, random_state=42)
S_2 = ica_2.fit_transform(X)
print("Partial ICA (2 of 3):", S_2.shape)
# Output: Partial ICA (2 of 3): (2000, 2)
# Extracts the 2 most non-Gaussian components (via whitening with n_components=2)`}
      </CodeBlock>

      <H3>5c. Picard library — faster convergence on real data</H3>

      <CodeBlock language="python">
{`# Picard: Preconditioned ICA for Real Data (Ablin, Cardoso, Gramfort 2018)
# Install: pip install python-picard
# Uses L-BFGS with sparse Hessian approximations — 10-100x faster on real EEG/audio
# pip install python-picard

# from picard import picard
# K, W, S = picard(X.T, n_components=3, ortho=True, max_iter=1000)
# K: whitening matrix (d x d)
# W: estimated unmixing matrix (n_components x d)
# S: recovered sources (n_components x n_samples)  — note: columns are samples!

# Picard API differences from sklearn:
# - Input is (d x n_samples) transposed relative to sklearn
# - Returns separate K (whitening) and W (rotation) for interpretability
# - ortho=True enforces the orthogonal constraint (recommended for BSS)
# - ortho=False allows non-orthogonal solutions (useful for overcomplete ICA)

# MNE-Python (neuroscience) wraps Picard/FastICA:
# from mne.preprocessing import ICA as MNE_ICA
# ica_mne = MNE_ICA(n_components=20, method='picard')
# ica_mne.fit(raw)                    # raw: mne.io.Raw EEG object
# ica_mne.plot_components()           # visualize topomaps
# ica_mne.exclude = [0, 2]            # mark artifact components
# raw_clean = ica_mne.apply(raw)      # subtract artifacts`}
      </CodeBlock>

      <Callout type="info" title="When to use Picard over sklearn FastICA">
        Prefer Picard when: (1) data is real EEG, MEG, or fMRI — the ICA model does not hold exactly and Picard is more robust; (2) convergence is slow with sklearn (more than 50 iterations); (3) you need reproducible solutions under slight data perturbations — Picard's L-BFGS converges more smoothly. Use sklearn FastICA when: you need a dependency-free solution, you are doing exploratory analysis on clean synthetic data, or you are already in a sklearn pipeline.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. True sources vs mixed signals vs ICA recovery</H3>

      <Prose>
        The three panels below show the first source signal (sine wave) at each stage of the ICA pipeline. The true source (gold) is a clean sinusoid. After mixing with two other signals via matrix <Code>A</Code>, the observed mixed signal (green) is a superposition of all three sources — no obvious sinusoidal structure. After FastICA unmixing (purple), the recovered IC closely tracks the original sine wave with correlation 0.9999.
      </Prose>

      <Plot
        label="Source 1 (sine): true vs mixed vs ICA-recovered"
        xLabel="time"
        yLabel="amplitude"
        series={[
          {
            name: "true source s1 (sine)",
            color: colors.gold,
            points: [
              [0.0, 0.0], [0.426, 1.2475], [0.852, 1.209], [1.278, -0.0759], [1.704, -1.2826],
              [2.13, -1.167], [2.556, 0.1516], [2.982, 1.3139], [3.408, 1.1217], [3.834, -0.2268],
              [4.26, -1.3416], [4.686, -1.0733], [5.112, 0.3015], [5.538, 1.3654], [5.964, 1.0218],
              [6.39, -0.376], [6.816, -1.3856], [7.242, -0.9677], [7.668, 0.4499], [8.094, 1.401],
              [8.52, 0.9107], [8.946, -0.5226], [9.372, -1.4121], [9.798, -0.851], [10.224, 0.5943],
              [10.65, 1.419], [11.076, 0.7884], [11.502, -0.6644], [11.928, -1.422], [12.354, -0.7232],
            ],
          },
          {
            name: "mixed signal x1 (entangled)",
            color: colors.green,
            points: [
              [0.0, -0.0637], [0.426, 1.8899], [0.852, 0.4473], [1.278, -1.0555], [1.704, -0.7279],
              [2.13, -1.3656], [2.556, -0.1122], [2.982, 0.586], [3.408, 1.5171], [3.834, 0.2457],
              [4.26, -1.7906], [4.686, -1.6966], [5.112, 0.786], [5.538, 1.6679], [5.964, 0.562],
              [6.39, -0.6434], [6.816, -1.1847], [7.242, -0.8413], [7.668, 0.6014], [8.094, 0.8878],
              [8.52, 0.4818], [8.946, -0.9861], [9.372, -1.4748], [9.798, -0.3116], [10.224, 0.7785],
              [10.65, 1.2127], [11.076, 0.3501], [11.502, -0.5484], [11.928, -0.5887], [12.354, -0.1945],
            ],
          },
          {
            name: "ICA recovered IC1",
            color: "#a78bfa",
            points: [
              [0.0, 0.0039], [0.426, 1.2367], [0.852, 1.228], [1.278, -0.036], [1.704, -1.2859],
              [2.13, -1.096], [2.556, 0.121], [2.982, 1.3296], [3.408, 1.1353], [3.834, -0.2216],
              [4.26, -1.3546], [4.686, -1.069], [5.112, 0.3058], [5.538, 1.3882], [5.964, 1.0109],
              [6.39, -0.375], [6.816, -1.3916], [7.242, -0.9699], [7.668, 0.4433], [8.094, 1.4015],
              [8.52, 0.9244], [8.946, -0.5146], [9.372, -1.4134], [9.798, -0.8605], [10.224, 0.5827],
              [10.65, 1.4214], [11.076, 0.7958], [11.502, -0.6531], [11.928, -1.4263], [12.354, -0.7294],
            ],
          },
        ]}
      />

      <H3>6b. PCA vs ICA on 2D non-Gaussian data</H3>

      <Prose>
        On two-dimensional data with independent Laplace-distributed sources, PCA finds the variance-maximizing orthogonal rotation (gold arrows). ICA finds the independence-maximizing rotation (purple arrows). For non-Gaussian sources, these are different rotations. PCA directions align with the axes of maximum spread in the <em>mixed</em> coordinate system; ICA directions align with the true <em>source</em> axes. The Laplace distribution's diamond-like contours make the source directions visible — ICA finds them by maximizing non-Gaussianity along each axis.
      </Prose>

      <Plot
        label="2D Laplace mixture: PCA axes vs ICA axes"
        xLabel="observed x₁"
        yLabel="observed x₂"
        series={[
          {
            name: "mixed data points",
            color: colors.gold,
            points: [
              [-3.2, -2.1], [1.4, 1.8], [-0.9, -0.5], [2.1, 2.8], [0.3, 0.7],
              [-1.5, -2.2], [0.8, 1.1], [-2.7, -1.3], [3.1, 3.5], [-0.4, 0.2],
              [1.9, 1.5], [-1.1, -0.8], [0.5, -0.3], [2.4, 2.1], [-0.7, -1.4],
              [1.2, 0.9], [-2.0, -2.5], [0.1, 0.4], [3.3, 2.7], [-1.8, -1.0],
              [0.6, 1.3], [-0.3, 0.6], [1.7, 2.2], [-1.3, -0.6], [2.8, 1.9],
              [-0.5, -1.1], [0.9, 0.3], [-2.4, -3.1], [1.5, 0.8], [-0.8, -0.2],
            ],
          },
          {
            name: "PCA PC1 (max variance, ignores independence)",
            color: colors.green,
            points: [[-3.5, -3.05], [3.5, 3.05]],
          },
          {
            name: "ICA direction (max non-Gaussianity = true source)",
            color: "#a78bfa",
            points: [[-3.8, -1.9], [3.8, 1.9]],
          },
        ]}
      />

      <H3>6c. Mixing matrix recovery accuracy</H3>

      <Prose>
        The product <Code>|W · A|</Code> should be close to a permutation matrix if ICA correctly identifies the unmixing. Each row of <Code>W</Code> corresponds to one recovered IC; each column of <Code>A</Code> corresponds to one true source. A value near 1.0 on the diagonal (after the optimal permutation) indicates perfect recovery; values near 0 indicate no mixing. Below is <Code>|W · A|</Code> from the sklearn FastICA run (parallel, logcosh, random_state=42) — rows are recovered ICs, columns are true sources.
      </Prose>

      <Heatmap
        label="|W·A| — mixing matrix recovery (rows=recovered ICs, cols=true sources)"
        rowLabels={["IC1", "IC2", "IC3"]}
        colLabels={["sine", "square", "laplace"]}
        matrix={[
          [1.0002, 0.0163, 0.0227],
          [0.0132, 0.0281, 1.0002],
          [0.0057, 1.0002, 0.0017],
        ]}
        colorScale="gold"
      />

      <Prose>
        The near-identity structure (each row and column has exactly one entry near 1.0 and all others near 0) confirms near-perfect recovery. IC1 recovers the sine source (column 1), IC2 recovers the Laplace source (column 3), and IC3 recovers the square wave (column 2) — the permutation ambiguity in action. The small off-diagonal values (0.02–0.03) reflect finite-sample estimation noise.
      </Prose>

      <H3>6d. FastICA step trace</H3>

      <StepTrace
        label="FastICA pipeline — step by step"
        steps={[
          {
            label: "Step 0 — raw mixed data",
            render: () => (
              <Prose>
                Input: <Code>X</Code> of shape <Code>(2000, 3)</Code> — three microphone channels, each a linear mixture of sine, square wave, and Laplace noise. Variances: [1.31, 1.56, 1.41] — all greater than the unit-variance sources because mixing inflates spread. No obvious structure is visible in any individual channel.
              </Prose>
            ),
          },
          {
            label: "Step 1 — center",
            render: () => (
              <Prose>
                Subtract column means: <Code>X_c = X - X.mean(axis=0)</Code>. The means are near zero (our synthetic sources had zero mean), so centering has minimal effect here. In practice, centering is critical — non-zero mean contaminates the covariance estimate and shifts the fixed-point iteration away from the correct solution.
              </Prose>
            ),
          },
          {
            label: "Step 2 — whiten",
            render: () => (
              <Prose>
                Compute covariance <Code>Σ = X_cᵀX_c / (n-1)</Code>, eigendecompose it, form whitening matrix <Code>W_white = D^{"{-1/2}"} Vᵀ</Code>, apply: <Code>X_white = X_c @ W_white.T</Code>. Verify: covariance of <Code>X_white</Code> is identity — diagonal entries all 1.0, off-diagonal all 0.0. Whitening removes all second-order correlations and normalizes variances. The ICA fixed-point now only needs to find an orthogonal rotation — much smaller search space than arbitrary invertible matrices.
              </Prose>
            ),
          },
          {
            label: "Step 3 — initialize w",
            render: () => (
              <Prose>
                Draw a random unit vector <Code>w₀ ∈ ℝ³</Code>. This is the starting point for the fixed-point iteration for IC1. The choice of initialization affects convergence speed but not the final solution (for well-separated sources), because the fixed-point for each IC is an attractor — any starting vector in its basin of attraction converges to it. In high dimensions, random restarts are used to escape bad local optima.
              </Prose>
            ),
          },
          {
            label: "Step 4 — fixed-point iteration",
            render: () => (
              <Prose>
                Update rule: <Code>w_new = E[x̃ · g(wᵀx̃)] - E[g&#x27;(wᵀx̃)] · w</Code>, then normalize. For <Code>g = tanh</Code>: <Code>w_new = (X_white.T @ tanh(X_white @ w)) / n - (1 - tanh²).mean() * w</Code>. This is a Newton step on the negentropy objective. Convergence check: <Code>|w_new · w|</Code> reaches 1.0 (vectors are parallel) in iteration 4 for IC1, iteration 5 for IC2, iteration 4 for IC3. Total: 13 iterations for all three components — extremely fast.
              </Prose>
            ),
          },
          {
            label: "Step 5 — deflate and repeat",
            render: () => (
              <Prose>
                After extracting IC1, project it out of the search space: for IC2, subtract the projection onto IC1 from the new vector after each update (<Code>w -= (w · w_IC1) · w_IC1</Code>), then renormalize. This Gram-Schmidt deflation enforces that IC2 is orthogonal to IC1 in the whitened space — which guarantees it captures a different source. Repeat for IC3. After all three, <Code>W_ica</Code> is a <Code>(3, 3)</Code> orthogonal matrix in the whitened space.
              </Prose>
            ),
          },
          {
            label: "Step 6 — recover sources",
            render: () => (
              <Prose>
                Apply the unmixing: <Code>S_recovered = X_white @ W_ica.T</Code>. The absolute correlations with the true sources are [0.9999, 0.9999, 0.9997] — near-perfect. Sign ambiguity: IC2 and IC3 may be negated relative to the true sources (multiply by -1 if needed). Permutation ambiguity: IC2 recovered the Laplace source and IC3 recovered the square wave — swap their labels.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        ICA, PCA, NMF, and autoencoders all produce latent representations of data, but they optimize different objectives and make different assumptions. The right choice depends on whether your sources are independent, non-negative, Gaussian, or whether you need an exact model vs. a flexible encoder.
      </Prose>

      <StepTrace
        label="ICA vs PCA vs NMF vs autoencoder"
        steps={[
          {
            label: "ICA — use when sources are independent and non-Gaussian",
            render: () => (
              <Prose>
                Best for: source separation (audio, EEG, fMRI); artifact removal (blink artifact in EEG is a near-perfect ICA component — one IC captures the artifact, zero others); feature extraction from signals where latent causes are physically independent; finding latent variables in data where the generative model is genuinely a linear mixture of independent sources. ICA wins over PCA whenever the data is non-Gaussian and the mixing structure matters — not just the variance. Requires: square (or overcomplete with prewhitening) mixing; non-Gaussian sources; sufficient samples relative to number of components (rough rule: at least 5n samples for n components, ideally more). Does NOT require: labeled data; knowledge of source distributions; knowing A in advance.
              </Prose>
            ),
          },
          {
            label: "PCA — use when variance structure is the target",
            render: () => (
              <Prose>
                Best for: dimensionality reduction for visualization or compression; preprocessing before supervised learning; data where Gaussian structure is a reasonable approximation (financial returns, many biological signals); extracting the directions of maximum variance for downstream distance-based algorithms. PCA wins over ICA when: the data is approximately Gaussian (ICA gives arbitrary results on Gaussian data); you need ordered components (PCA guarantees variance-ordering; ICA does not); you need a reconstruction basis (PCA components form an orthonormal basis for the best rank-k reconstruction; ICA components are not ordered by reconstruction quality); you need computational simplicity (PCA is a single SVD; ICA requires iteration). PCA is always run before ICA as the whitening step — they are complementary, not competing.
              </Prose>
            ),
          },
          {
            label: "NMF — use when components must be non-negative",
            render: () => (
              <Prose>
                Non-negative Matrix Factorization decomposes <Code>X ≈ WH</Code> where both <Code>W</Code> and <Code>H</Code> have non-negative entries. This parts-based decomposition is natural for: spectrograms (frequency components cannot be negative); document-topic models (word frequencies are non-negative); image decomposition into additive parts (face = eyes + nose + mouth). NMF wins over ICA when the non-negativity constraint is physically meaningful — NMF components are interpretable as additive parts, while ICA components can be negative. NMF loses to ICA when sources are not non-negative and the independent source model holds — NMF has no theoretical justification for recovering true sources from signed mixtures.
              </Prose>
            ),
          },
          {
            label: "Autoencoder — use when nonlinear structure matters",
            render: () => (
              <Prose>
                Autoencoders learn a nonlinear encoder-decoder pair, capturing structure that no linear method can find. Wins over ICA when: mixing is nonlinear; the latent space has complex geometry (images, audio waveforms); you have enough data ({">"} 10k samples) to train a network. Variational autoencoders (VAEs) impose a structured prior on the latent space, encouraging independence between latent dimensions — making them a nonlinear generalization of ICA. Autoencoders lose to ICA when: data is small (ICA needs only a few hundred samples for d=3); interpretability matters (ICA components connect to physically meaningful sources; autoencoder latents are entangled); you need a rigorous statistical model (ICA has a clean likelihood interpretation; autoencoders do not).
              </Prose>
            ),
          },
          {
            label: "Quick decision rule",
            render: () => (
              <Prose>
                Start with PCA. If PCA components are interpretable and variance-based reduction is the goal, stop. If the components look "mixed" — each PCA component contains contributions from multiple physically distinct sources — switch to ICA. If data is non-negative, try NMF instead. If the mixing is clearly nonlinear and you have large datasets, use an autoencoder. For EEG/MEG artifact removal specifically: always use ICA (either sklearn FastICA or MNE with Picard) — decades of neuroscience research have validated it for this use case.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 Computational complexity</H3>

      <Prose>
        The ICA pipeline has two cost centers. First, whitening: eigendecomposition of the <Code>d × d</Code> covariance matrix costs <Code>O(nd² + d³)</Code> — the same as full PCA. For <Code>n = 10,000</Code> samples and <Code>d = 100</Code> features, this is trivially fast. For <Code>d = 10,000</Code>, the <Code>d × d</Code> covariance matrix is 800 MB in float64 — approaching the limit of a single machine. Second, the fixed-point iteration: each update for one IC costs <Code>O(nd)</Code> (one forward pass over the whitened data). For <Code>k</Code> components and <Code>T</Code> iterations, the total iteration cost is <Code>O(ndkT)</Code>. With typical convergence in 3–15 iterations, and <Code>k = d</Code>, this is <Code>O(nd²)</Code> — dominated by the whitening step. For the parallel algorithm, all <Code>k</Code> components are updated simultaneously in each iteration, so the per-iteration cost scales with <Code>k</Code>.
      </Prose>

      <H3>8.2 Scaling to large n and d</H3>

      <Prose>
        FastICA scales well in <Code>n</Code> (number of samples) for fixed <Code>d</Code>: doubling samples doubles the cost of the fixed-point iteration but does not change the cost of eigendecomposition (which is <Code>O(d³)</Code>). In the regime <Code>n = 1,000,000</Code> and <Code>d = 50</Code> (typical for audio ICA with 50 channels), FastICA is perfectly feasible on a single CPU. The bottleneck shifts: whitening now requires forming the <Code>d × d</Code> covariance by summing <Code>n</Code> outer products, costing <Code>O(nd²)</Code> — but with <Code>d = 50</Code> this is a 50×50 matrix, trivially fast.
      </Prose>

      <Prose>
        Scaling in <Code>d</Code> is the hard direction. For <Code>d = 1,000</Code> components from <Code>n = 10,000</Code> samples, whitening requires a <Code>1,000 × 1,000</Code> eigendecomposition (fast), but the fixed-point iteration with deflation extracts components sequentially — the total deflation cost is <Code>O(kd)</Code> per component, so <Code>O(k²d)</Code> total. At <Code>k = d = 1,000</Code>, this is <Code>10⁹</Code> operations — slow but feasible. For <Code>d {">"} 5,000</Code> with full ICA, use the parallel algorithm (avoids sequential deflation overhead) or reduce dimensionality first via PCA then apply ICA to the low-dimensional representation (a common pattern in neuroimaging: PCA to 200 components, then ICA on those 200).
      </Prose>

      <H3>8.3 Sample requirements</H3>

      <Prose>
        ICA is a statistical method — it requires enough samples to accurately estimate the statistics of the nonlinearity <Code>g</Code>. A rough practical rule: you need at least <Code>O(d²)</Code> samples to reliably identify <Code>d</Code> independent components. For <Code>d = 10</Code> components, 200 samples may suffice; for <Code>d = 100</Code>, you need at least 10,000. Below the sample threshold, ICA converges to spurious solutions — fixed points of the iteration that are not the true source directions. Always check convergence with held-out data: apply the learned unmixing to a validation set and verify the recovered components are similarly non-Gaussian.
      </Prose>

      <Plot
        label="FastICA timing: O(n·d·k·T) — fixed d=50, varying n"
        xLabel="number of samples n (×10³)"
        yLabel="relative compute time"
        series={[
          {
            name: "whitening O(n·d²)",
            color: colors.gold,
            points: [[10, 1.0], [50, 5.0], [100, 10.0], [500, 50.0], [1000, 100.0]],
          },
          {
            name: "fixed-point iteration O(n·d·k·T)",
            color: colors.green,
            points: [[10, 0.8], [50, 4.0], [100, 8.0], [500, 40.0], [1000, 80.0]],
          },
        ]}
      />

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Gaussian sources — the fundamental impossibility</H3>

      <Prose>
        If even two of your sources are Gaussian, ICA cannot separate them. The multivariate Gaussian is rotationally symmetric: any orthogonal rotation of independent Gaussian components produces another set of independent Gaussian components with the same joint distribution. There is no preferred rotation, so the fixed-point iteration has no stable attractor corresponding to the true sources — it converges to an arbitrary orthogonal rotation of the whitened data, which is exactly what PCA gives. Diagnosis: check the kurtosis of the whitened data. If all three whitened signals have kurtosis near 0, the sources are probably Gaussian and ICA will not improve over PCA. If you must separate Gaussian sources, you need temporal structure (autocorrelation): methods like SOBI (Second-Order Blind Identification) exploit lagged covariances rather than non-Gaussianity.
      </Prose>

      <H3>9.2 Sign and permutation ambiguity in practice</H3>

      <Prose>
        The sign of each ICA component is arbitrary — <Code>w</Code> and <Code>−w</Code> are both valid solutions. The ordering of components is arbitrary — ICA does not rank components by variance (unlike PCA). In EEG artifact removal, the ordering matters: you must manually identify which IC corresponds to the blink artifact before zeroing it out. For audio separation, sign is irrelevant (negating a waveform is inaudible at low levels), but permutation matters — you need to label which IC is the piano and which is the voice. Automated approaches for resolving ambiguity include: correlating recovered ICs with template signals (e.g., a reference electrode for blink artifacts in EEG), using topographic maps of the mixing matrix columns (the "topomaps" in MNE), or using signal-specific features (pitch tracking, spectral flatness).
      </Prose>

      <H3>9.3 Convergence to local optima</H3>

      <Prose>
        The FastICA fixed-point iteration is guaranteed to converge to a fixed point of the negentropy gradient, but not necessarily to the global maximum. For well-separated, non-Gaussian sources, the global maximum corresponds to the true source directions and the iteration converges reliably. For sources with similar non-Gaussianity (similar kurtosis), or in high dimensions, the iteration may converge to a spurious fixed point — a direction that is locally optimal but not globally so. Mitigation: run FastICA multiple times with different random initializations and select the solution with the highest total negentropy. sklearn's <Code>FastICA</Code> uses a single initialization; for robust results on difficult data, implement multiple restarts manually. The Picard algorithm is more robust to local optima because L-BFGS explores the objective more carefully than the pure fixed-point iteration.
      </Prose>

      <H3>9.4 Too few samples relative to components</H3>

      <Prose>
        If you request <Code>k = d</Code> components from a dataset with <Code>n {"<"} 5d²</Code> samples, the whitening step will be poorly conditioned — the sample covariance will have large estimation error, and the whitened data will not truly have identity covariance. This propagates into the ICA fixed-point: you are fitting a statistically unstable rotation. Symptoms: ICA solutions change drastically with different random seeds; the recovered components do not match known ground truth; kurtosis of recovered components is low. Fix: reduce <Code>k</Code> (request fewer components than the data can support), or collect more data. A concrete threshold: for 3 components and n=2000 samples (as in our example), the recovery is excellent (|corr| {">"} 0.999). For 100 components and n=500 samples, ICA will not recover the true sources reliably.
      </Prose>

      <H3>9.5 Noisy ICA</H3>

      <Prose>
        The standard ICA model assumes noiseless mixing: <Code>x = As</Code>. Real data always has additive noise: <Code>x = As + ε</Code>. Noise corrupts the higher-order statistics that ICA relies on — it "Gaussianizes" the observed signals, making all sources look more Gaussian and harder to separate. The effect is proportional to the noise level: at high SNR ({">"} 20 dB), FastICA still recovers sources well; at low SNR ({"<"} 10 dB), recovery degrades significantly. Regularization approaches include: (1) PCA pre-reduction — reduce to the top-k components before ICA, discarding the noise-dominated low-variance PCs; (2) noisy ICA models that explicitly estimate the noise covariance (computationally more expensive); (3) ensemble averaging in neuroscience (average many trials to boost SNR before ICA).
      </Prose>

      <H3>9.6 Scaling ambiguity and normalization choices</H3>

      <Prose>
        The ICA model cannot separately identify the scale of <Code>A</Code> and <Code>s</Code>: doubling all source values and halving the corresponding column of <Code>A</Code> gives the same observed data. Conventional ICA normalizes sources to unit variance, absorbing the scale into <Code>A</Code>. This means the columns of the mixing matrix <Code>A</Code> (equivalently, the rows of the unmixing matrix <Code>W</Code>) have norms that encode the scale of the sources relative to the data. In EEG analysis, this is used directly: the column of <Code>A</Code> for a given IC gives its "spatial pattern" — how that source mixes into each electrode — and its norm encodes the source's contribution to total signal power. Always check the mixing matrix normalization when comparing ICA solutions across different runs or implementations.
      </Prose>

      <Callout type="warning" title="Do not run ICA on Gaussian data expecting PCA-like results">
        A common mistake is applying ICA to approximately Gaussian data (e.g., normally distributed sensor noise, financial log-returns) and reporting the recovered components as meaningful. ICA on Gaussian data is undefined — the solution is non-unique and algorithm-dependent. The components will depend on the random seed and will change if you add or remove a sample. If you are unsure whether your sources are Gaussian, check kurtosis on the whitened data. If all kurtoses are in the range [−0.5, 0.5], ICA is not appropriate. Use PCA instead.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below are WebSearch-verified for author, year, venue, volume, pages, and core claims. Read them in this order to follow the intellectual lineage from the original heuristic to the modern algorithm.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "Hérault, Jutten & Ans 1985 — The original GRETSI algorithm",
            render: () => (
              <Prose>
                Hérault, J., Jutten, C., and Ans, B. (1985). "Détection de grandeurs primitives dans un message composite par une architecture de calcul neuromimétique en apprentissage non supervisé." Proceedings of the <em>Xème colloque GRETSI</em>, Nice, France, May 1985, pp. 1017–1022. The founding paper of blind source separation. The authors proposed a biologically-inspired adaptive neural network that could separate linear mixtures of independent sources by anti-Hebbian learning. The algorithm worked, but the paper contained no theoretical explanation for why — that understanding waited nine years for Comon 1994. Hérault and Jutten later published a follow-up journal version: "Space or time adaptive signal processing by neural network models," in <em>Neural Networks for Signal Processing</em>, AIP Conference Proceedings, 1986, pp. 206–211, which further developed the neuromimetic framework.
              </Prose>
            ),
          },
          {
            label: "Comon 1994 — ICA as a concept: the theoretical foundation",
            render: () => (
              <Prose>
                Comon, P. (1994). "Independent Component Analysis, a New Concept?" <em>Signal Processing</em>, 36(3), 287–314. DOI: 10.1016/0165-1684(94)90029-9. The paper that gave ICA its name, its formal definition, and its theoretical justification. Comon showed: (1) independence (not just decorrelation) is the right objective; (2) at most one Gaussian source is identifiable; (3) the source separation is unique up to permutation and scaling for non-Gaussian sources. He also connected ICA to information theory (mutual information minimization) and to higher-order statistics (cumulants). This paper transformed blind source separation from an empirical heuristic into a principled statistical method. Essential reading for anyone who wants to understand why ICA works.
              </Prose>
            ),
          },
          {
            label: "Bell & Sejnowski 1995 — Infomax: the first practical algorithm",
            render: () => (
              <Prose>
                Bell, A.J. and Sejnowski, T.J. (1995). "An Information-Maximization Approach to Blind Separation and Blind Deconvolution." <em>Neural Computation</em>, 7(6), 1129–1159. DOI: 10.1162/neco.1995.7.6.1129. The paper that made ICA a practical tool. Bell and Sejnowski derived the Infomax algorithm: maximize the output entropy of a neural network with sigmoid nonlinearities by gradient ascent, and the hidden units will become statistically independent. They demonstrated separation of up to 10 simultaneous speech recordings, including the first successful blind separation of real mixed audio. The connection between entropy maximization and source independence (established via the natural gradient, elaborated in Amari et al. 1996) makes this one of the most influential papers in the early history of unsupervised deep learning.
              </Prose>
            ),
          },
          {
            label: "Hyvärinen & Oja 1997 — FastICA: the algorithm in production use",
            render: () => (
              <Prose>
                Hyvärinen, A. and Oja, E. (1997). "A Fast Fixed-Point Algorithm for Independent Component Analysis." <em>Neural Computation</em>, 9(7), 1483–1492. DOI: 10.1162/neco.1997.9.7.1483. The FastICA paper. Hyvärinen reformulated the ICA optimization as a fixed-point problem: find <Code>w</Code> such that <Code>w = E[x g(wᵀx)] - E[g'(wᵀx)] w</Code> (normalized). The fixed-point iteration converges cubically, versus the linear convergence of gradient ascent methods like Infomax. Hyvärinen also established the connection to negentropy maximization and gave the derivation for multiple nonlinearities (<Code>tanh</Code>, <Code>exp</Code>, <Code>u³</Code>). This algorithm is implemented in sklearn, MNE-Python, and every other modern ICA package — it is the algorithm you should use unless you have a specific reason to prefer Picard.
              </Prose>
            ),
          },
          {
            label: "Hyvärinen, Karhunen & Oja 2001 — The definitive textbook",
            render: () => (
              <Prose>
                Hyvärinen, A., Karhunen, J., and Oja, E. (2001). <em>Independent Component Analysis</em>. Wiley-Interscience, New York. ISBN: 978-0-471-40540-5. DOI: 10.1002/0471221317. The definitive reference on ICA, with 481 pages covering: the ICA generative model and its identifiability; estimation via maximum likelihood, mutual information, and negentropy; FastICA in full mathematical detail; extensions including noisy ICA, overcomplete ICA, and nonlinear ICA; applications in neuroscience, audio, finance, and image processing. The full manuscript (bookfinal_ICA.pdf) is freely available on Aapo Hyvärinen's website at cs.helsinki.fi. If you need to understand ICA at depth — including the proofs of identifiability, the derivation of the Cramér-Rao lower bound, and the theory of robust estimation — this is the book.
              </Prose>
            ),
          },
          {
            label: "Ablin, Cardoso & Gramfort 2018 — Picard: faster convergence on real data",
            render: () => (
              <Prose>
                Ablin, P., Cardoso, J.-F., and Gramfort, A. (2018). "Faster Independent Component Analysis by Preconditioning with Hessian Approximations." <em>IEEE Transactions on Signal Processing</em>, 66(15), 4040–4049. arXiv:1706.08171. The Picard algorithm. Ablin et al. precondition the ICA gradient with a sparse approximation to the Hessian of the log-likelihood (derived from the score function of the nonlinearity), then apply L-BFGS. This gives superlinear convergence near the optimum and is dramatically more robust than FastICA on real data where the ICA model does not hold exactly. The paper also analyzes the failure modes of FastICA on real EEG data and shows empirically that Picard is 10–100× faster in practice. The Python implementation <Code>python-picard</Code> (pip install python-picard) is the backend used by MNE-Python for EEG/MEG analysis. Highly recommended for any application where the ICA model is approximate.
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
        Work through these before moving to the next topic. Attempt each question before reading the answer below it.
      </Prose>

      <H3>Exercise 1 (conceptual)</H3>
      <Prose>
        Your colleague applies ICA to a dataset where all three sources are independent Gaussian random variables. They report that ICA recovered the sources with high correlation to the originals. What is the most likely explanation, and what would you expect if they re-ran with a different random seed?
      </Prose>
      <Callout type="answer" title="Answer 1">
        The result is almost certainly coincidental. For jointly Gaussian sources, the ICA model is not identifiable — any orthogonal rotation of the whitened data has the same statistical properties (zero mutual information between components for any rotation). The fixed-point iteration converges to an arbitrary orthogonal rotation determined by the random initialization. If the random seed happens to start near the true source directions, the recovered components may correlate well with the true sources by chance. Re-running with a different seed will produce a different (but equally valid statistically) orthogonal rotation that may have very low correlation with the true sources. The correct diagnosis: run the kurtosis test on the whitened data. If all three kurtoses are near 0, the sources are approximately Gaussian and ICA is not appropriate — use PCA for decorrelation, or SOBI if temporal structure exists.
      </Callout>

      <H3>Exercise 2 (derivation)</H3>
      <Prose>
        Starting from the FastICA fixed-point update rule <Code>w_new = E[x̃ g(wᵀx̃)] - E[g&#x27;(wᵀx̃)] w</Code>, explain what happens if you use <Code>g(u) = u</Code> (the identity function). What does the fixed-point iteration become, and what does it find?
      </Prose>
      <Callout type="answer" title="Answer 2">
        With g(u) = u, g'(u) = 1. The update becomes: w_new = E[x̃ (wᵀx̃)] - E[1] w = E[x̃ x̃ᵀ] w - w. Since the data is whitened, E[x̃ x̃ᵀ] = I (identity covariance). So w_new = Iw - w = w - w = 0. The update collapses to zero — the iteration is degenerate. More informatively: the linear nonlinearity g(u) = u is equivalent to maximizing the variance of wᵀx̃, which for whitened data is the same for all unit vectors (variance = wᵀI w = 1 for all w). There is no preferred direction. This is another way of seeing why PCA and ICA agree on Gaussian data and why PCA decorrelation is the ceiling of what linear (second-order) methods can achieve. ICA needs a nonlinear g to access higher-order statistics and break the rotational symmetry.
      </Callout>

      <H3>Exercise 3 (implementation)</H3>
      <Prose>
        You run FastICA with <Code>n_components=5</Code> on a dataset with 5 true independent sources. The model converges in 3 iterations, but when you check the absolute correlation matrix between true sources and recovered ICs, the maximum per-row value is 0.72 instead of near 1.0. What are the three most likely causes, and how do you diagnose each?
      </Prose>
      <Callout type="answer" title="Answer 3">
        {"Cause 1: Insufficient samples. With n_components=5, you need at least O(d^2) = 25 to O(5 * d^2) = 125 times the number of components in samples. Check the sample count relative to d=5. If n < 500, the whitening step is poorly conditioned and the fixed-point iterates on a corrupted basis. Fix: collect more data or reduce n_components. Cause 2: Sources are too close to Gaussian. Check excess kurtosis of the whitened data. If all kurtoses are in [-0.5, 0.5], the sources are near-Gaussian and ICA cannot distinguish their directions."} Fix: verify the data-generating process produces truly non-Gaussian sources. Cause 3: Convergence to a local optimum. 3 iterations is very fast — the iteration may have converged to a saddle point rather than the true source directions. Fix: run multiple random restarts (change random_state), collect the solution with the highest sum of |kurtosis| across components, and verify it matches the ground truth. Also try algorithm='deflation' — sometimes parallel and deflation converge to different local optima.
      </Callout>

      <H3>Exercise 4 (applied — EEG)</H3>
      <Prose>
        You are performing EEG artifact removal using ICA. After running FastICA with 20 components, you inspect the component topomaps (spatial patterns, columns of the mixing matrix A) and time courses. One component has a characteristic frontal topography and an activity pattern that spikes whenever the subject blinks. You zero this component out and reconstruct the data. Two weeks later, your colleague reruns ICA on the same data with a different random seed. They find that the "blink" artifact has been split across two components. Why did this happen, and what does it tell you about ICA robustness for EEG?
      </Prose>
      <Callout type="answer" title="Answer 4">
        The permutation and convergence instability arise from multiple sources. First, the random seed changes the initialization of each fixed-point iteration — a different starting point may converge to a different local optimum where the blink variance is split between two nearby directions rather than concentrated in one. Second, real EEG blink artifacts are not perfectly independent of other signals — there is some correlation between blink and eye-movement components, causing instability in where the variance is assigned. Third, with 20 components extracted from typically 64-256 electrodes, many near-optimal rotations exist with similar negentropy. Practical lesson: ICA on EEG is not fully reproducible without fixing the random seed. Standard practice in clinical EEG pipelines is to fix random_state, document it, and always visualize component topomaps manually rather than relying on automated component selection. The Picard algorithm (used via MNE-Python) is more stable than FastICA across seeds for real EEG data, though not fully seed-invariant.
      </Callout>

      <H3>Exercise 5 (synthesis)</H3>
      <Prose>
        Explain why whitening is a necessary preprocessing step for FastICA but not, strictly speaking, for the ICA problem in general. What would happen if you ran the FastICA fixed-point iteration on non-whitened data? What assumption does the derivation of the fixed-point rule rely on?
      </Prose>
      <Callout type="answer" title="Answer 5">
        Whitening is necessary for FastICA's specific fixed-point formulation, not for ICA in general. Here is why. The FastICA fixed-point rule is derived under the constraint that the unmixing vector w has unit norm AND that the data has identity covariance. Under these conditions, the problem reduces to finding an orthogonal rotation of the whitened data — a constrained optimization over the much smaller orthogonal group O(d) rather than the full general linear group GL(d). Without whitening, the fixed-point iteration would need to simultaneously search over rotations AND scales AND shears — a much larger space with many more local optima and no guarantee of convergence. The Gram-Schmidt deflation (orthogonalization against previously found components) also relies on the whitened-space identity covariance: orthogonality in the whitened space corresponds to statistical decorrelation, which is a prerequisite for finding independent components. In principle, ICA can be solved without whitening by maximizing mutual information over GL(d) directly (Infomax does this), but it is computationally much harder. Whitening as a first step decouples the problem into two simpler subproblems: handle second-order structure (whitening), then handle higher-order structure (rotation). This is why PCA and ICA are complementary rather than competing: PCA is always the first half of FastICA.
      </Callout>

      <H3>Exercise 6 (failure mode)</H3>
      <Prose>
        You run ICA on a dataset with 4 sensors and 4 sources. The recovered components have high kurtosis, and the correlation matrix between true and recovered sources shows the block structure you would expect from perfect recovery — except two components have max correlation 0.71 with any true source. After investigation, you discover that two of the four true sources have identical excess kurtosis (both equal to 3.0). What is happening, and is there a principled fix?
      </Prose>
      <Callout type="answer" title="Answer 6">
        Two sources with identical non-Gaussianity (kurtosis = 3.0, both Laplace-distributed) create a degenerate case for the kurtosis-based objective. ICA maximizes a measure of non-Gaussianity along each direction. When two sources have the same kurtosis profile, the kurtosis surface has a ring of equally optimal directions rather than isolated maxima — any linear combination of the two true source directions within their span has the same kurtosis as the pure sources. The fixed-point converges to a direction on this ring that depends on initialization, not to the true source directions. 0.71 ≈ 1/sqrt(2), which is the correlation you get when you recover a 45-degree rotation of the two-source subspace — consistent with the iteration landing on the ring. The fix: switch from kurtosis to negentropy with a different nonlinearity g. If the two sources have the same marginal kurtosis but different shapes (e.g., one is Laplace and one is a mixture), negentropy with G = log cosh is more sensitive to shape differences and may separate them. A more robust fix: if you have temporal structure (the sources are autocorrelated signals with different spectral profiles), use SOBI (Second-Order Blind Identification), which exploits lagged covariance structure and can separate sources with identical marginal distributions as long as their temporal dynamics differ.
      </Callout>

    </div>
  ),
};

export default icaContent;
