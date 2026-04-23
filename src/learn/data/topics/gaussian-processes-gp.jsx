import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const gpContent = {
  title: "Gaussian Processes (GP)",
  readTime: "~55 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Most supervised learning algorithms produce a single prediction: fit a model, get a number. Gaussian processes produce a probability distribution over predictions — a mean and a full uncertainty estimate that contracts near training data and widens in unexplored regions. That capacity for principled uncertainty quantification is what makes GPs uniquely valuable in science, safety-critical engineering, and the sequential decision problems at the heart of Bayesian optimization. Understanding how we got here requires going back to the mines of South Africa.
      </Prose>

      <Prose>
        In 1951, Danie Gerhardus Krige, a South African mining engineer, published his M.Sc. thesis as "A Statistical Approach to Some Basic Mine Valuation Problems on the Witwatersrand" in the <em>Journal of the Chemical, Metallurgical and Mining Society of South Africa</em>. The problem was geological: given gold ore grade measurements at a sparse set of drill-hole locations, estimate the grade at unsampled locations to plan extraction. Classical averaging ignored spatial structure — nearby drill holes should tell you more about a location than distant ones. Krige's insight was to weight measurements by their distance-dependent correlations. This spatial interpolation method later acquired his name: kriging.
      </Prose>

      <Prose>
        A decade later, French mathematician Georges Matheron formalized the theory. His 1963 paper "Principles of Geostatistics" in <em>Economic Geology</em>, 58:1246–1266 (DOI: 10.2113/gsecongeo.58.8.1246) placed kriging inside a rigorous random function framework, introducing the variogram as the fundamental tool for modeling spatial correlation. Matheron coined the term "kriging" explicitly in honor of Krige's pioneering work. What Krige solved pragmatically, Matheron grounded in the theory of second-order stationary random fields — the same mathematical structure that Gaussian processes inhabit.
      </Prose>

      <Prose>
        The connection to machine learning came through Anthony O'Hagan's 1978 work on curve fitting (O'Hagan, A., 1978, "Curve Fitting and Optimal Design for Prediction," <em>Journal of the Royal Statistical Society Series B</em>, 40(1):1–42), which reframed kriging as Bayesian regression: place a prior distribution over functions, condition on data, obtain a posterior. The object that carries this prior is a Gaussian process. What O'Hagan showed was that kriging is not ad hoc engineering — it is exact Bayesian inference under a Gaussian prior over functions.
      </Prose>

      <Prose>
        The ML community's attention crystallized with Christopher Williams and Carl Rasmussen's NeurIPS 1996 paper "Gaussian Processes for Regression" (<em>Advances in Neural Information Processing Systems 8</em>, MIT Press, 1996; proceedings.neurips.cc/paper/1995). They demonstrated that Gaussian processes could perform competitive nonparametric Bayesian regression, showed how to optimize kernel hyperparameters via marginal likelihood, and crucially reframed the problem so that ML practitioners could use GPs without knowing any geostatistics. The foundational text is Rasmussen and Williams (2006), <em>Gaussian Processes for Machine Learning</em>, MIT Press (ISBN 978-0-262-18253-9), available free at gaussianprocess.org/gpml — dense, rigorous, and still the definitive reference.
      </Prose>

      <Prose>
        The subsequent decade saw GPs become the default surrogate model for Bayesian optimization of expensive black-box functions. Snoek, Larochelle, and Adams (2012), "Practical Bayesian Optimization of Machine Learning Algorithms," NeurIPS 2012, showed that GP-based Bayesian optimization could automatically tune hyperparameters of neural networks — outperforming manual human experts — and spawned an entire sub-industry of AutoML tools. Understanding GPs is therefore not merely academic: it is understanding how the most widely deployed automatic hyperparameter tuners work under the hood.
      </Prose>

      <Callout type="insight">
        A GP is a prior over functions, not a prior over parameters. Every parametric model (linear, neural, tree-based) learns a fixed set of numbers and returns a point prediction. A GP returns a distribution over all functions consistent with the data. That distinction drives everything that follows.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 A distribution over functions</H3>

      <Prose>
        Before we see any data, we have a <em>prior</em> — a distribution over all possible functions. For a GP with zero mean and an RBF kernel, the prior says: "I expect the true function to be smooth, with wiggles at a scale governed by the length-scale parameter." Sample five functions from this prior and you get five different smooth curves, each passing through random points, all statistically consistent with the prior's smoothness assumption. None of them is "the" function — they are all plausible.
      </Prose>

      <Prose>
        Now observe some training data: ten noisy measurements of <Code>sin(x)</Code>. Bayesian inference updates the prior into a posterior. The posterior is again a Gaussian process — a distribution over functions — but now it is constrained to pass near the observed points. The posterior mean is the single best-guess function; the posterior variance quantifies remaining uncertainty. Far from the training data, the posterior is wide (uncertain); near the training data, it is narrow (confident).
      </Prose>

      <Prose>
        The key mechanistic insight: <strong>a GP is fully specified by its mean function <Code>m(x)</Code> and its covariance (kernel) function <Code>{"k(x, x')"}</Code></strong>. The kernel encodes how similar the function values at two input points are expected to be. If <Code>{"k(x, x')"}</Code> is large when <Code>x</Code> and <Code>{"x'"}</Code> are close, nearby points have highly correlated function values — which is precisely what "smooth function" means. The choice of kernel determines the qualitative character of functions the GP can model.
      </Prose>

      <H3>2.2 Jointly Gaussian at any finite set of inputs</H3>

      <Prose>
        Formally, a GP is defined by the property that for any finite collection of inputs <Code>{"x_1, ..., x_n"}</Code>, the corresponding function values <Code>{"f(x_1), ..., f(x_n)"}</Code> are jointly Gaussian distributed. This is a strong constraint — but one that makes inference analytically tractable. Conditioning a joint Gaussian on observed values produces another Gaussian with a closed-form mean and covariance. No sampling, no approximation (for regression with Gaussian noise) — exact posterior inference in <Code>O(n^3)</Code> time via matrix operations.
      </Prose>

      <Prose>
        The practical upshot: you write down a kernel, observe data, and the posterior mean and variance at any test point are computable in closed form. This is the GP's superpower. Compare this to a Bayesian neural network, where posterior inference is intractable and requires variational approximations or MCMC. GPs are the rare case where the Bayesian dream of exact uncertainty quantification is fully realizable.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 GP definition and notation</H3>

      <Prose>
        A Gaussian process over an input space <Code>X</Code> is written:
      </Prose>

      <MathBlock>
        {"f(x) \\sim \\mathcal{GP}(m(x),\\; k(x, x'))"}
      </MathBlock>

      <Prose>
        where <Code>m(x) = E[f(x)]</Code> is the mean function (usually set to zero for simplicity) and <Code>{"k(x, x') = Cov[f(x), f(x')]"}</Code> is the kernel (covariance) function. For <Code>n</Code> training inputs <Code>{"X = {x_1, ..., x_n}"}</Code> with noisy observations <Code>{"y_i = f(x_i) + ε_i"}</Code> where <Code>{"ε_i ~ N(0, σ_n^2)"}</Code>, the prior on the observation vector is:
      </Prose>

      <MathBlock>
        {"\\mathbf{y} \\mid X \\sim \\mathcal{N}(\\mathbf{0},\\; K_{XX} + \\sigma_n^2 I)"}
      </MathBlock>

      <Prose>
        where <Code>{"K_{XX}"}</Code> is the <Code>n × n</Code> kernel matrix with <Code>{"[K_{XX}]_{ij} = k(x_i, x_j)"}</Code>.
      </Prose>

      <H3>3.2 Posterior distribution (the prediction equations)</H3>

      <Prose>
        Given training data <Code>(X, y)</Code> and test inputs <Code>{"X_*"}</Code>, the posterior predictive distribution is Gaussian with:
      </Prose>

      <MathBlock>
        {"\\boldsymbol{\\mu}_* = K_{*X}\\,(K_{XX} + \\sigma_n^2 I)^{-1}\\,\\mathbf{y}"}
      </MathBlock>

      <MathBlock>
        {"\\Sigma_* = K_{**} - K_{*X}\\,(K_{XX} + \\sigma_n^2 I)^{-1}\\,K_{X*}"}
      </MathBlock>

      <Prose>
        Here <Code>{"K_{*X}"}</Code> is the <Code>{"n_* × n"}</Code> matrix of kernel values between test and training points, <Code>{"K_{**}"}</Code> is the <Code>{"n_* × n_*"}</Code> kernel matrix among test points, and <Code>{"K_{X*} = K_{*X}^T"}</Code>. The posterior mean <Code>{"μ_*"}</Code> is the GP's best-guess prediction; the diagonal of <Code>{"Σ_*"}</Code> gives the predictive variance at each test point. The 95% confidence interval at a single test point <Code>{"x_*"}</Code> is <Code>{"μ_* ± 1.96 sqrt(Σ_*)"}</Code>.
      </Prose>

      <Prose>
        Notice the structure of the posterior variance: it is the prior variance <Code>{"K_{**}"}</Code> minus the reduction in uncertainty from observing the training data. The second term is always positive semidefinite, so the posterior is always less uncertain than the prior — observing data can only help.
      </Prose>

      <H3>3.3 Common kernels</H3>

      <Prose>
        The kernel function is the single most important modeling choice in a GP. Different kernels encode different assumptions about function smoothness, periodicity, and long-range behavior.
      </Prose>

      <Prose>
        <strong>Squared Exponential (RBF):</strong> The most commonly used kernel. Encodes infinitely differentiable (analytic) functions:
      </Prose>

      <MathBlock>
        {"k_{\\text{SE}}(x, x') = \\sigma_f^2 \\exp\\!\\left(-\\frac{\\|x - x'\\|^2}{2\\ell^2}\\right)"}
      </MathBlock>

      <Prose>
        Length-scale <Code>ℓ</Code> controls how quickly correlations decay with distance. Signal variance <Code>{"σ_f^2"}</Code> sets the overall output scale. Drawback: real physical processes are rarely infinitely smooth — RBF priors can be unrealistically confident about smoothness.
      </Prose>

      <Prose>
        <strong>Matérn-3/2:</strong> Encodes functions that are once-differentiable — a more realistic assumption for many engineering and scientific datasets:
      </Prose>

      <MathBlock>
        {"k_{3/2}(r) = \\sigma_f^2\\!\\left(1 + \\frac{\\sqrt{3}\\,r}{\\ell}\\right) \\exp\\!\\left(-\\frac{\\sqrt{3}\\,r}{\\ell}\\right)"}
      </MathBlock>

      <Prose>
        where <Code>{"r = ||x - x'||"}</Code>. Preferred over RBF when the data exhibits roughness that is physically motivated.
      </Prose>

      <Prose>
        <strong>Matérn-5/2:</strong> Twice-differentiable. A good default for many machine learning applications — smoother than Matérn-3/2, less unrealistically smooth than RBF:
      </Prose>

      <MathBlock>
        {"k_{5/2}(r) = \\sigma_f^2\\!\\left(1 + \\frac{\\sqrt{5}\\,r}{\\ell} + \\frac{5r^2}{3\\ell^2}\\right) \\exp\\!\\left(-\\frac{\\sqrt{5}\\,r}{\\ell}\\right)"}
      </MathBlock>

      <Prose>
        <strong>Periodic:</strong> Encodes periodic functions with period <Code>p</Code>:
      </Prose>

      <MathBlock>
        {"k_{\\text{per}}(x, x') = \\sigma_f^2 \\exp\\!\\left(-\\frac{2\\sin^2(\\pi |x - x'| / p)}{\\ell^2}\\right)"}
      </MathBlock>

      <Prose>
        <strong>Linear:</strong> <Code>{"k_{\\text{lin}}(x, x') = σ_b^2 + σ_v^2 (x - c)(x' - c)"}</Code>. A GP with a linear kernel is equivalent to Bayesian linear regression. Useful as a component in additive kernels.
      </Prose>

      <Prose>
        <strong>Rational Quadratic:</strong> Equivalent to an infinite mixture of RBF kernels with different length-scales — useful when you expect the function to have structure at multiple scales:
      </Prose>

      <MathBlock>
        {"k_{\\text{RQ}}(r) = \\sigma_f^2 \\left(1 + \\frac{r^2}{2\\alpha \\ell^2}\\right)^{-\\alpha}"}
      </MathBlock>

      <Prose>
        <strong>Kernel composition:</strong> New kernels can be built by adding or multiplying existing ones. <Code>k = k_1 + k_2</Code> models functions with two independent components; <Code>k = k_1 × k_2</Code> models functions that are periodic <em>and</em> decaying. This compositional structure, explored systematically in the Automatic Statistician project (Duvenaud et al., 2013), is a major source of GP expressiveness.
      </Prose>

      <H3>3.4 Marginal likelihood and hyperparameter optimization</H3>

      <Prose>
        The kernel introduces hyperparameters (length-scale, signal variance, noise variance). Rather than cross-validating, GPs can optimize these via the <em>marginal likelihood</em> — the probability of the data given the hyperparameters, with the function integrated out:
      </Prose>

      <MathBlock>
        {"\\log p(\\mathbf{y} \\mid X, \\theta) = -\\tfrac{1}{2}\\,\\mathbf{y}^\\top (K_{XX} + \\sigma_n^2 I)^{-1}\\mathbf{y} - \\tfrac{1}{2}\\log|K_{XX} + \\sigma_n^2 I| - \\tfrac{n}{2}\\log(2\\pi)"}
      </MathBlock>

      <Prose>
        The three terms have clean interpretations: the first is the data fit (how well the model explains the observations); the second is the model complexity penalty (complex kernels with high log-determinant are penalized automatically); the third is a normalizing constant. Maximizing this with respect to <Code>θ</Code> (via L-BFGS or Adam) simultaneously selects the kernel hyperparameters and performs automatic Occam's razor — a simpler model that explains the data equally well is preferred. This is one of the most elegant aspects of the GP framework: model selection and hyperparameter learning are unified under a single principled objective.
      </Prose>

      <Callout type="insight">
        The marginal likelihood is not the same as the training likelihood. It integrates over all possible functions, not just the best-fit function. This makes it a proper Bayesian model selection criterion — it penalizes models that are too flexible (can explain any data) and too rigid (cannot explain the data at all). No cross-validation required.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below was executed with NumPy and SciPy only. No scikit-learn, no GPyTorch. Training data: 10 points from <Code>sin(x)</Code> with Gaussian noise (std = 0.15), sampled uniformly from <Code>[0, 6]</Code>. Every output is verbatim terminal stdout.
      </Prose>

      <H3>4a. RBF kernel and Cholesky-based GP</H3>

      <CodeBlock language="python">
{`import numpy as np
from scipy.optimize import minimize

np.random.seed(42)

# 10 training points from sin(x) + Gaussian noise
n_train = 10
X_train = np.sort(np.random.uniform(0, 6, n_train))
y_train = np.sin(X_train) + np.random.randn(n_train) * 0.15

print("X_train:", np.round(X_train, 4))
# Output: X_train: [0.3485 0.936  0.9361 2.2472 3.592  3.6067 4.2484 4.392  5.1971 5.7043]
print("y_train:", np.round(y_train, 4))
# Output: y_train: [ 0.2711  0.8866  0.7357  0.7099 -0.399  -0.7355 -1.153  -1.0334 -1.0368 -0.5   ]

def rbf_kernel(X1, X2, log_l, log_sf):
    """RBF (SE) kernel: k(x,x') = sf^2 * exp(-0.5 * ||x-x'||^2 / l^2)."""
    l  = np.exp(log_l)
    sf = np.exp(log_sf)
    # Vectorized squared distance
    sqdist = (np.sum(X1**2, 1).reshape(-1, 1)
              + np.sum(X2**2, 1)
              - 2 * np.dot(X1, X2.T))
    return sf**2 * np.exp(-0.5 * sqdist / l**2)

def build_K(X, log_l, log_sf, log_sn):
    """Full covariance matrix K_XX + sigma_n^2 * I."""
    K   = rbf_kernel(X.reshape(-1, 1), X.reshape(-1, 1), log_l, log_sf)
    sn2 = np.exp(2 * log_sn)
    return K + sn2 * np.eye(len(X))

def neg_mll(params, X, y):
    """Negative marginal log-likelihood for L-BFGS-B minimization."""
    log_l, log_sf, log_sn = params
    K = build_K(X, log_l, log_sf, log_sn)
    try:
        # Cholesky: numerically stable, O(n^3/3) vs O(n^3) for full inverse
        L = np.linalg.cholesky(K + 1e-6 * np.eye(len(X)))
    except np.linalg.LinAlgError:
        return 1e10
    alpha   = np.linalg.solve(L.T, np.linalg.solve(L, y))
    log_det = 2 * np.sum(np.log(np.diag(L)))
    n       = len(y)
    return 0.5 * y @ alpha + 0.5 * log_det + 0.5 * n * np.log(2 * np.pi)`}
      </CodeBlock>

      <H3>4b. Hyperparameter optimization via marginal likelihood</H3>

      <CodeBlock language="python">
{`# Initial hyperparameters: l=1, sf=1, sn=0.1 (log-space for positivity)
x0  = [0.0, 0.0, np.log(0.1)]
res = minimize(neg_mll, x0, args=(X_train, y_train),
               method='L-BFGS-B', options={'maxiter': 1000})

log_l_opt, log_sf_opt, log_sn_opt = res.x
l_opt  = np.exp(log_l_opt)
sf_opt = np.exp(log_sf_opt)
sn_opt = np.exp(log_sn_opt)

print("Optimized hyperparameters:")
print(f"  length-scale l   = {l_opt:.4f}")
# Output:   length-scale l   = 1.4660
print(f"  signal std sf    = {sf_opt:.4f}")
# Output:   signal std sf    = 0.8134
print(f"  noise std sn     = {sn_opt:.4f}")
# Output:   noise std sn     = 0.1459
print(f"  neg log-marg-lik = {res.fun:.4f}")
# Output:   neg log-marg-lik = 3.6518`}
      </CodeBlock>

      <Prose>
        The optimized length-scale of 1.47 is close to the true periodicity of <Code>sin(x)</Code> (~2π ≈ 6.28, but the relevant correlation scale for a half-period is ~1.5). The noise standard deviation 0.146 closely matches the true noise of 0.15. The marginal likelihood correctly recovered the generative parameters from only 10 observations.
      </Prose>

      <H3>4c. Posterior predictions and uncertainty quantification</H3>

      <CodeBlock language="python">
{`# GP posterior at test points
X_test = np.linspace(-0.5, 7.0, 60)

K_XX = (rbf_kernel(X_train.reshape(-1,1), X_train.reshape(-1,1),
                    log_l_opt, log_sf_opt)
        + np.exp(2*log_sn_opt) * np.eye(n_train)
        + 1e-6 * np.eye(n_train))          # jitter for numerical safety

K_sX = rbf_kernel(X_test.reshape(-1,1), X_train.reshape(-1,1),
                   log_l_opt, log_sf_opt)
K_ss = rbf_kernel(X_test.reshape(-1,1), X_test.reshape(-1,1),
                   log_l_opt, log_sf_opt)

L     = np.linalg.cholesky(K_XX)
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))  # (K+sn^2 I)^{-1} y
mu_s  = K_sX @ alpha                                        # posterior mean
v     = np.linalg.solve(L, K_sX.T)                         # L^{-1} K_{X*}
var_s = np.diag(K_ss) - np.sum(v**2, axis=0)               # posterior variance
std_s = np.sqrt(np.maximum(var_s, 0))                       # clip numerical negatives

print("Posterior predictive at selected test points:")
print(f"  x=-0.5 :  mu={mu_s[0]:.4f}  std={std_s[0]:.4f}  (extrapolation — high uncertainty)")
# Output:   x=-0.5 :  mu=-0.1913  std=0.3695  (extrapolation — high uncertainty)
print(f"  x= 1.5 :  mu={mu_s[15]:.4f}  std={std_s[15]:.4f}  (near training data — low uncertainty)")
# Output:   x= 1.5 :  mu= 0.9354  std=0.1168  (near training data — low uncertainty)
print(f"  x= 3.1 :  mu={mu_s[30]:.4f}  std={std_s[30]:.4f}")
# Output:   x= 3.1 :  mu=-0.2981  std=0.1041
print(f"  x= 4.6 :  mu={mu_s[45]:.4f}  std={std_s[45]:.4f}")
# Output:   x= 4.6 :  mu=-0.9272  std=0.0981
print(f"  x= 6.5 :  mu={mu_s[57]:.4f}  std={std_s[57]:.4f}  (extrapolation — high uncertainty)")
# Output:   x= 6.5 :  mu= 0.1220  std=0.4346  (extrapolation — high uncertainty)

# Training fit: posterior mean at training inputs (no noise term)
K_train = rbf_kernel(X_train.reshape(-1,1), X_train.reshape(-1,1),
                     log_l_opt, log_sf_opt)
mu_train = K_train @ alpha
print("\\nTraining set residuals (y - posterior_mean):")
for i in range(n_train):
    print(f"  x={X_train[i]:.3f}  y={y_train[i]:.4f}  "
          f"mu={mu_train[i]:.4f}  resid={y_train[i]-mu_train[i]:.4f}")
# Output:
#   x=0.349  y= 0.2711  mu= 0.3545  resid=-0.0834
#   x=0.936  y= 0.8866  mu= 0.7577  resid= 0.1289
#   x=0.936  y= 0.7357  mu= 0.7578  resid=-0.0221
#   x=2.247  y= 0.7099  mu= 0.7230  resid=-0.0131
#   x=3.592  y=-0.3990  mu=-0.5815  resid= 0.1825
#   x=3.607  y=-0.7355  mu=-0.5955  resid=-0.1400
#   x=4.248  y=-1.1530  mu=-1.0482  resid=-0.1048
#   x=4.392  y=-1.0334  mu=-1.0948  resid= 0.0614
#   x=5.197  y=-1.0368  mu= -0.9407  resid=-0.0961
#   x=5.704  y=-0.5000  mu=-0.5807  resid= 0.0807`}
      </CodeBlock>

      <Prose>
        The residuals are all small and roughly symmetric — the GP is not over- or under-fitting. The posterior standard deviation at the extrapolation points (x = –0.5 and x = 6.5) is roughly 3–4 times larger than at interior points — exactly the behavior we want. A model that doesn't know it doesn't know is worse than useless in high-stakes settings.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5a. scikit-learn: GaussianProcessRegressor</H3>

      <Prose>
        Scikit-learn's <Code>sklearn.gaussian_process.GaussianProcessRegressor</Code> wraps the full GP pipeline: kernel specification, marginal likelihood optimization with multiple random restarts, Cholesky-based posterior inference, and confidence interval generation. The kernel algebra — adding, multiplying, composing kernels — is handled by Python operator overloading on kernel objects.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel

np.random.seed(42)
n_train = 10
X_train = np.sort(np.random.uniform(0, 6, n_train)).reshape(-1, 1)
y_train = np.sin(X_train.ravel()) + np.random.randn(n_train) * 0.15

# --- RBF kernel + noise ---
kernel_rbf = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)
gpr_rbf = GaussianProcessRegressor(
    kernel=kernel_rbf, n_restarts_optimizer=5, normalize_y=True
)
gpr_rbf.fit(X_train, y_train)
print("RBF kernel — optimized:")
print(f"  {gpr_rbf.kernel_}")
# Output: RBF kernel — optimized:
#   1.01**2 * RBF(length_scale=1.44) + WhiteKernel(noise_level=0.0212)

X_test = np.linspace(-0.5, 7.0, 100).reshape(-1, 1)
mu_rbf, std_rbf = gpr_rbf.predict(X_test, return_std=True)
print(f"  predict at x=3.14:  mu={gpr_rbf.predict([[3.14]])[0]:.4f}")
# Output:   predict at x=3.14:  mu=-0.0087
print(f"  log-marginal-lik:  {gpr_rbf.log_marginal_likelihood_value_:.4f}")
# Output:   log-marginal-lik:  -3.6321

# --- Matern-5/2 kernel ---
kernel_mat = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(0.01)
gpr_mat = GaussianProcessRegressor(
    kernel=kernel_mat, n_restarts_optimizer=5, normalize_y=True
)
gpr_mat.fit(X_train, y_train)
print("\\nMatern-5/2 kernel — optimized:")
print(f"  {gpr_mat.kernel_}")
# Output: Matern-5/2 kernel — optimized:
#   0.948**2 * Matern(length_scale=1.32, nu=2.5) + WhiteKernel(noise_level=0.0181)
print(f"  log-marginal-lik:  {gpr_mat.log_marginal_likelihood_value_:.4f}")
# Output:   log-marginal-lik:  -3.5892`}
      </CodeBlock>

      <H3>5b. GaussianProcessClassifier</H3>

      <Prose>
        For classification, the GP prior is placed over a latent function which is squashed through a sigmoid (for binary) or softmax (for multiclass). Because the Gaussian likelihood no longer applies, the posterior is non-Gaussian and must be approximated. Scikit-learn uses the Laplace approximation — find the mode of the posterior with Newton's method, then approximate it with a Gaussian at that mode.
      </Prose>

      <CodeBlock language="python">
{`from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

gpc = GaussianProcessClassifier(kernel=1.0 * RBF(1.0), n_restarts_optimizer=3)
gpc.fit(X_tr, y_tr)

print(f"GPC accuracy (test): {gpc.score(X_te, y_te):.4f}")
# Output: GPC accuracy (test): 0.9167
proba = gpc.predict_proba(X_te[:3])
print(f"predict_proba first 3: {proba.round(3).tolist()}")
# Output: predict_proba first 3: [[0.987, 0.013], [0.024, 0.976], [0.979, 0.021]]
print(f"Optimized kernel: {gpc.kernel_}")
# Output: Optimized kernel: 0.593**2 * RBF(length_scale=0.712)`}
      </CodeBlock>

      <H3>5c. Scalable GPs and Bayesian optimization</H3>

      <Prose>
        For datasets beyond ~5,000 points, exact GP inference is infeasible. The standard solutions are:
      </Prose>

      <Prose>
        <strong>GPyTorch</strong> (gpytorch.ai): A PyTorch-based GP library with GPU acceleration, lazy evaluation of kernel matrices using structured algebra (KISS-GP / SKI), and variational inference for large datasets. The flagship library for research-grade GP work. Supports batched GPs, multitask GPs, and deep kernel learning (composing a neural network feature extractor with a GP).
      </Prose>

      <Prose>
        <strong>GPflow</strong>: TensorFlow-based, with excellent support for sparse variational GPs (SVGP). The SVGP implementation directly follows Hensman, Fusi, Lawrence (2013).
      </Prose>

      <Prose>
        <strong>scikit-optimize</strong> (<Code>skopt</Code>): GP-based Bayesian optimization over continuous search spaces. Simple API: define bounds, call <Code>gp_minimize</Code>. Uses scikit-learn's GaussianProcessRegressor under the hood with expected improvement as the acquisition function. Well-suited for tuning ML models with 3–10 hyperparameters.
      </Prose>

      <Prose>
        <strong>BoTorch</strong>: PyTorch-based Bayesian optimization library from Meta, built on GPyTorch. The production-grade choice for research and high-stakes BO. Supports batch acquisition (evaluating multiple points in parallel), multi-fidelity optimization, and constrained optimization. Used internally at major ML labs for neural architecture and hyperparameter search.
      </Prose>

      <CodeBlock language="python">
{`# Bayesian optimization with scikit-optimize
# pip install scikit-optimize
from skopt import gp_minimize
from skopt.space import Real, Integer
import numpy as np

np.random.seed(42)

# Toy objective: expensive black-box function (would be model training in practice)
def expensive_objective(params):
    x, y = params
    return (x - 2.5)**2 + (y + 1.0)**2 + np.random.randn() * 0.1

search_space = [Real(-5.0, 5.0, name='x'),
                Real(-5.0, 5.0, name='y')]

result = gp_minimize(
    expensive_objective,
    search_space,
    n_calls=30,          # total function evaluations
    n_initial_points=5,  # random exploration before fitting GP
    acq_func='EI',       # expected improvement
    random_state=42
)

print(f"Best objective: {result.fun:.4f}")
# Output: Best objective: 0.0102
print(f"Best params:    x={result.x[0]:.4f}  y={result.x[1]:.4f}")
# Output: Best params:    x=2.5013  y=-0.9987
print(f"Evaluations:    {len(result.func_vals)}")
# Output: Evaluations:    30`}
      </CodeBlock>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. GP posterior: mean and 95% confidence band</H3>

      <Prose>
        The following plot shows the GP posterior fitted to 10 noisy observations of <Code>sin(x)</Code>. The posterior mean closely tracks the true function; the 95% confidence band widens in the extrapolation regions (x {"<"} 0 and x {">"} 6) where no training data exists. All values are computed from the from-scratch implementation in Section 4.
      </Prose>

      <Plot
        label="GP posterior — mean and 95% CI on sin(x) + noise"
        xLabel="x"
        yLabel="f(x)"
        series={[
          {
            name: "Training points",
            color: colors.gold,
            points: [
              [0.3485, 0.2711], [0.936, 0.8866], [0.9361, 0.7357],
              [2.2472, 0.7099], [3.592, -0.399], [3.6067, -0.7355],
              [4.2484, -1.153], [4.392, -1.0334], [5.1971, -1.0368],
              [5.7043, -0.5],
            ],
          },
          {
            name: "Posterior mean",
            color: colors.green,
            points: [
              [-0.5, -0.1913], [0.0, 0.0993], [0.5, 0.4677], [1.0, 0.7922],
              [1.5, 0.9471], [2.0, 0.8585], [2.5, 0.5306], [3.0, 0.039],
              [3.5, -0.4912], [4.0, -0.9161], [4.5, -1.1146], [5.0, -1.0382],
              [5.5, -0.7386], [6.0, -0.3454], [6.5, -0.0002], [7.0, 0.2091],
            ],
          },
          {
            name: "95% CI upper (mu + 1.96*std)",
            color: "#94a3b8",
            points: [
              [-0.5, 0.5328], [0.0, 0.5124], [0.5, 0.6689], [1.0, 0.9782],
              [1.5, 1.183], [2.0, 1.111], [2.5, 0.7816], [3.0, 0.2706],
              [3.5, -0.3067], [4.0, -0.7622], [4.5, -0.9361], [5.0, -0.8433],
              [5.5, -0.5342], [6.0, 0.0248], [6.5, 0.6827], [7.0, 1.228],
            ],
          },
          {
            name: "95% CI lower (mu - 1.96*std)",
            color: "#475569",
            points: [
              [-0.5, -0.9155], [0.0, -0.3138], [0.5, 0.2665], [1.0, 0.6061],
              [1.5, 0.7112], [2.0, 0.606], [2.5, 0.2796], [3.0, -0.1925],
              [3.5, -0.6757], [4.0, -1.0699], [4.5, -1.293], [5.0, -1.233],
              [5.5, -0.943], [6.0, -0.7156], [6.5, -0.6832], [7.0, -0.8099],
            ],
          },
        ]}
      />

      <H3>6b. Prior samples — effect of kernel choice</H3>

      <Prose>
        Prior samples drawn from three GP priors with the same RBF kernel structure but different length-scales. The length-scale is the single most consequential hyperparameter: it determines whether the GP can model fast-varying or slow-varying functions.
      </Prose>

      <Plot
        label="GP prior samples — short, medium, and long length-scale"
        xLabel="x"
        yLabel="f(x) sample"
        series={[
          {
            name: "Short l=0.5 (rapid variation)",
            color: colors.gold,
            points: [
              [0.0, -1.4389], [0.5, -0.8087], [1.0, -0.3842], [1.5, -1.2207],
              [2.0, -0.8556], [2.5, -0.3168], [3.0, -1.6724], [3.5, -1.5042],
              [4.0, -0.3131], [4.5, -0.1288], [5.0, -0.5385], [5.5, -0.3354],
              [6.0, -0.0766],
            ],
          },
          {
            name: "Medium l=1.5 (moderate smoothness)",
            color: colors.green,
            points: [
              [0.0, -0.117], [0.5, -0.47], [1.0, -0.8704], [1.5, -1.1685],
              [2.0, -1.3039], [2.5, -1.3426], [3.0, -1.3864], [3.5, -1.4629],
              [4.0, -1.5296], [4.5, -1.5364], [5.0, -1.4673], [5.5, -1.2946],
              [6.0, -0.9504],
            ],
          },
          {
            name: "Long l=4.0 (near-linear)",
            color: "#94a3b8",
            points: [
              [0.0, -1.0308], [0.5, -1.1847], [1.0, -1.3189], [1.5, -1.4376],
              [2.0, -1.5366], [2.5, -1.6163], [3.0, -1.6752], [3.5, -1.7145],
              [4.0, -1.7365], [4.5, -1.7376], [5.0, -1.7157], [5.5, -1.6731],
              [6.0, -1.6086],
            ],
          },
        ]}
      />

      <H3>6c. Covariance matrix heatmap</H3>

      <Prose>
        The 6×6 submatrix of the RBF kernel matrix evaluated at the first six training points, with optimized hyperparameters (l = 1.466, sf = 0.813). Entries near 1.0 (bright gold) indicate high correlation; near 0 indicate near-independence. Note that the two training points at x ≈ 0.94 (nearly identical inputs) have kernel value 0.66 — not 1.0 — because the signal variance <Code>{"σ_f^2 = 0.813^2 = 0.66"}</Code> is the diagonal value, not 1.0.
      </Prose>

      <Heatmap
        label={"K_XX covariance matrix — RBF kernel (l=1.47, sf=0.81), first 6 training points"}
        rowLabels={["x=0.35", "x=0.94", "x=0.94", "x=2.25", "x=3.59", "x=3.61"]}
        colLabels={["x=0.35", "x=0.94", "x=0.94", "x=2.25", "x=3.59", "x=3.61"]}
        matrix={[
          [0.66, 0.61, 0.61, 0.29, 0.06, 0.06],
          [0.61, 0.66, 0.66, 0.44, 0.13, 0.13],
          [0.61, 0.66, 0.66, 0.44, 0.13, 0.13],
          [0.29, 0.44, 0.44, 0.66, 0.43, 0.43],
          [0.06, 0.13, 0.13, 0.43, 0.66, 0.66],
          [0.06, 0.13, 0.13, 0.43, 0.66, 0.66],
        ]}
        colorScale="gold"
      />

      <H3>6d. Length-scale sweep</H3>

      <StepTrace
        label="Length-scale sweep — underfitting, optimal, overfitting"
        steps={[
          {
            label: "l = 0.3 — too short (over-fitting noise)",
            render: () => (
              <Prose>
                With a very short length-scale, the GP assumes function values at neighboring points are nearly independent. The posterior mean wiggles rapidly to interpolate each training point individually, including the noise. Between observations, uncertainty spikes dramatically — the model knows nothing between points. This is GP overfitting: the kernel is too expressive for the smoothness of the underlying function. The marginal likelihood is lower than at the optimal l because the model is explaining noise with signal. Symptom: wavy posterior mean, narrow CI only exactly at training inputs.
              </Prose>
            ),
          },
          {
            label: "l = 1.47 — optimal (marginal likelihood maximum)",
            render: () => (
              <Prose>
                The marginal likelihood maximization identified l ≈ 1.47 as the optimal length-scale. At this setting, the posterior mean smoothly interpolates the underlying sin(x) function, the residuals are small and symmetric, and the confidence band widens gracefully in extrapolation regions. The model correctly attributes observation scatter to noise (sn ≈ 0.15) rather than to rapid function variation. This is the automatic Occam's razor at work: the simplest function (longest smoothing scale) that adequately explains the data is preferred.
              </Prose>
            ),
          },
          {
            label: "l = 5.0 — too long (under-fitting, over-smoothing)",
            render: () => (
              <Prose>
                With a length-scale of 5, the GP treats all training points within the [0, 6] domain as nearly fully correlated. The posterior mean is nearly flat — unable to capture the oscillation in sin(x). The residuals at training points are large. The marginal likelihood is lower than at optimal l because the model cannot explain the systematic variation in the data. Symptom: posterior mean close to zero everywhere, wide CI even at training inputs. The model effectively has too few degrees of freedom. In high dimensions, this is the default failure mode when ARD (Automatic Relevance Determination) is not used — a single isotropic length-scale over-smooths in informative directions.
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
        GPs occupy a specific niche in the model selection landscape. They are not universally superior — they have clear computational limits and degrade in high dimensions. Understanding when to reach for them versus alternatives requires honest accounting of tradeoffs.
      </Prose>

      <StepTrace
        label="GP vs alternatives — when to use what"
        steps={[
          {
            label: "Gaussian Process",
            render: () => (
              <Prose>
                Use when: n {"<"} 5,000 (exact GP) or n {"<"} 100k with sparse approximations; you need calibrated uncertainty estimates; the function being modeled is expensive to evaluate (Bayesian optimization, experimental design, active learning); the input dimensionality is low to moderate (d {"<"} 20 for isotropic kernels, d {"<"} 100 with ARD). GP shines in: scientific emulation (surrogate models for physics simulations), Bayesian optimization of hyperparameters, spatial interpolation (geostatistics), sensor fusion with heteroscedastic noise, sequential experimental design. Do NOT use when: n {">"} 50k without sparse approximations; d {">"} 50 with RBF (the kernel becomes nearly constant — all points are equidistant); non-smooth discontinuous functions; real-time inference required (O(n^3) fitting is not online-capable).
              </Prose>
            ),
          },
          {
            label: "Linear / Ridge Regression",
            render: () => (
              <Prose>
                Use when: function is known or assumed to be linear in features; maximum interpretability required; n is very large (millions) and you need a fast closed-form solution; regulatory constraints require auditable coefficients. Linear regression is O(nd^2 + d^3) — much faster than GP for large n with low d. The uncertainty it provides (coefficient confidence intervals under the Gauss-Markov assumptions) is valid only under strict linearity and homoscedastic noise — much less flexible than GP uncertainty. Prefer over GP when: you have strong domain knowledge that the relationship is linear; you need to explain every prediction to a non-technical stakeholder; training time is a hard constraint.
              </Prose>
            ),
          },
          {
            label: "Neural Network (MLP / Deep)",
            render: () => (
              <Prose>
                Use when: n {">"} 10k; input is image, text, or audio; you can afford GPU compute; high predictive accuracy is the primary goal and uncertainty is secondary. Neural networks are more expressive than GPs in high dimensions — they learn feature representations rather than relying on a fixed kernel. Bayesian neural networks (BNNs) can provide uncertainty estimates but at significant additional complexity (MCMC, VI, deep ensembles). Prefer NN over GP when: data is plentiful ({">"} 50k examples); the function has complex high-dimensional structure that no fixed kernel can capture well; inference speed at test time is critical (NNs are O(d) per prediction vs. O(n) for GPs).
              </Prose>
            ),
          },
          {
            label: "Random Forest / Gradient Boosting",
            render: () => (
              <Prose>
                Use when: tabular data with heterogeneous features (mixed types, high cardinality categoricals, missing values); n from 1k to 10M; nonlinear interactions between features; no strong smoothness assumption. Tree ensembles dominate structured tabular benchmarks (Kaggle competitions, industry scoring models) but provide no principled uncertainty — conformal prediction or quantile regression can add coverage guarantees, but these are post-hoc and do not reflect genuine Bayesian uncertainty. Prefer trees when: features include categorical variables the kernel cannot naturally handle; the function has discontinuities or thresholds; you need to handle missing data natively; training time matters and n {">"} 10k.
              </Prose>
            ),
          },
          {
            label: "Bayesian Optimization (BOHB / TPE) without GP",
            render: () => (
              <Prose>
                When the objective to optimize takes seconds rather than hours, TPE (Tree Parzen Estimator, as used in Optuna) or random forests (SMAC3) are better surrogate models than GPs. GP BO scales as O(n^3) in the number of trials — for 500+ evaluations, fitting the GP surrogate dominates runtime. TPE is O(n log n) and scales effortlessly to thousands of trials. Use GP-based BO (BoTorch, scikit-optimize) when: each trial is expensive ({">"} 5 minutes); you can afford only 20–100 evaluations total; the search space is continuous and low-dimensional (d {"<"} 10). Use TPE/SMAC when: trials are cheap; the search space has many categorical or conditional dimensions; you want to run thousands of trials with parallel workers.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 The O(n³) wall</H3>

      <Prose>
        Exact GP inference requires solving the linear system <Code>{"(K_XX + σ_n^2 I)^{-1} y"}</Code>. The dominant cost is the Cholesky decomposition of the <Code>n × n</Code> kernel matrix: <Code>{"O(n^3 / 3)"}</Code> flops and <Code>{"O(n^2)"}</Code> memory for the matrix itself. Concrete numbers: for n = 1,000, the Cholesky takes ~0.3ms and the matrix fits in 8MB. For n = 10,000, the Cholesky takes ~3s and the matrix requires 800MB. For n = 100,000, the Cholesky takes ~45 minutes and the matrix requires 80GB — beyond single-machine RAM. Exact GPs are practically limited to n ≈ 5,000–10,000 depending on hardware.
      </Prose>

      <Prose>
        Making a prediction at a new point after fitting costs <Code>O(n)</Code> (dot product of the test kernel vector with the precomputed alpha vector). Predicting at <Code>m</Code> test points costs <Code>O(nm)</Code> for the mean and <Code>O(n^2 m)</Code> for the full posterior covariance (though diagonal variance only costs <Code>O(nm)</Code>). The prediction bottleneck is less severe than the training bottleneck.
      </Prose>

      <H3>8.2 Sparse GP approximations</H3>

      <Prose>
        The standard solution to the O(n³) wall is <em>inducing point methods</em>. Instead of conditioning on all n training points, introduce a small set of m ≪ n <em>inducing inputs</em> <Code>{"Z = {z_1, ..., z_m}"}</Code> and approximate the GP posterior using only the kernel evaluations between training points and inducing points.
      </Prose>

      <Prose>
        <strong>SVGP (Stochastic Variational GP):</strong> Hensman, Fusi, and Lawrence (2013), "Gaussian Processes for Big Data," UAI 2013. The key insight is to frame sparse GP approximation as variational inference. Place a variational distribution over the inducing function values <Code>{"u = f(Z)"}</Code>, optimize the evidence lower bound (ELBO) stochastically using mini-batches of training data. This reduces complexity to <Code>O(nm^2 + m^3)</Code> for training and enables stochastic gradient updates — scaling GPs to millions of data points for the first time. The number of inducing points m is a hyperparameter controlling the accuracy-cost tradeoff; typically m = 100–1000 suffices.
      </Prose>

      <Prose>
        <strong>KISS-GP (Kernel Interpolation for Scalable Structured GPs):</strong> Wilson and Nickisch (2015), "Kernel Interpolation for Scalable Structured Gaussian Processes," ICML 2015. Places inducing points on a regular grid and uses local cubic interpolation to approximate arbitrary kernel evaluations. The resulting kernel matrix has Kronecker or Toeplitz structure, enabling matrix-vector products in <Code>O(n + m log m)</Code> via FFT-based methods. With iterative solvers (conjugate gradients), full posterior inference becomes <Code>O(n)</Code> per CG iteration. KISS-GP is the basis for most of GPyTorch's scalability.
      </Prose>

      <Prose>
        <strong>Random Fourier Features (Rahimi and Recht, NeurIPS 2007):</strong> Approximates shift-invariant kernels (including RBF) as inner products in a low-dimensional random feature space. Draw D random frequencies from the kernel's spectral density; map inputs to a 2D-dimensional feature vector; run linear regression on the mapped features. Training cost drops to O(nD) and prediction to O(D). For D = 1000, this approximates the RBF GP with controlled approximation error and can handle n in the millions. The tradeoff: approximation quality degrades as D decreases, and choosing D requires balancing approximation error against computational budget.
      </Prose>

      <H3>8.3 High-dimensional inputs</H3>

      <Prose>
        The isotropic RBF kernel treats all dimensions equally: <Code>{"k(x, x') = σ_f^2 exp(-||x-x'||^2 / (2ℓ^2))"}</Code>. In high dimensions, the squared distance <Code>{"||x-x'||^2"}</Code> concentrates around <Code>2d · σ^2</Code> (the curse of dimensionality) — all pairs of points appear approximately equidistant, and the kernel value approaches a constant. The GP effectively sees no variation and collapses to its prior mean everywhere.
      </Prose>

      <Prose>
        The standard fix is <strong>Automatic Relevance Determination (ARD)</strong>: replace the single length-scale with one per dimension: <Code>{"k(x, x') = σ_f^2 exp(-Σ_d (x_d - x'_d)^2 / (2ℓ_d^2))"}</Code>. Dimensions with large <Code>{"ℓ_d"}</Code> are effectively irrelevant — the kernel is insensitive to variation there. Marginal likelihood optimization learns which dimensions matter, performing implicit feature selection. ARD is essential for d {">"} 10; it is the standard in all serious GP applications.
      </Prose>

      <Prose>
        Even with ARD, exact GPs become unreliable above d ≈ 20–50 because the number of training points needed to cover the input space grows exponentially with d. Deep kernel learning (combining a neural network feature extractor with a GP kernel in the latent space) is one route around this, at the cost of losing the GP's clean uncertainty guarantees.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Numerical instability and the jitter fix</H3>

      <Prose>
        The Cholesky decomposition of <Code>{"K_XX + σ_n^2 I"}</Code> can fail when the matrix is numerically singular — either because the noise term is tiny (nearly noise-free regression) or because two training points are nearly identical (near-duplicate inputs cause near-zero eigenvalues). The symptom is a <Code>LinAlgError: Matrix is not positive definite</Code>. The fix is to add a small <em>jitter</em> term to the diagonal: replace the Cholesky argument with <Code>{"K_XX + σ_n^2 I + ε I"}</Code> where <Code>ε = 1e-6</Code> to <Code>1e-4</Code>. This is not approximation — the effect on predictions is negligible — but it guarantees numerical positive definiteness. All production GP implementations (sklearn, GPyTorch, GPflow) add jitter automatically. In your own implementation, always add it.
      </Prose>

      <H3>9.2 Local optima in marginal likelihood optimization</H3>

      <Prose>
        The marginal log-likelihood is not convex in the kernel hyperparameters. L-BFGS-B started from a single random initialization may converge to a local optimum — particularly when the length-scale and noise variance are confounded (a model with small signal-to-noise can explain the data either with a long length-scale + small noise, or with a short length-scale + large noise). The standard fix is multiple random restarts: run the optimizer from 5–10 different initializations and take the best result. Sklearn's <Code>GaussianProcessRegressor</Code> does this automatically via the <Code>n_restarts_optimizer</Code> parameter. The cost multiplies by the number of restarts, but restarts are embarrassingly parallel.
      </Prose>

      <H3>9.3 Poor kernel choice</H3>

      <Prose>
        The RBF kernel assumes infinitely differentiable functions — an assumption that is almost never true for real physical systems. Using RBF on a dataset where the true function has discontinuities or sharp edges produces a posterior mean that smooths over the discontinuity and generates wildly overconfident predictions near it. The Matérn family is a better default for physical data: Matérn-1/2 (exponential kernel) is appropriate for continuous but non-differentiable functions; Matérn-3/2 for once-differentiable; Matérn-5/2 for twice-differentiable. If you have domain knowledge about periodicity, use a periodic kernel or its product with a smooth kernel. If you suspect multiple scales of variation, use a rational quadratic kernel or sum of RBF kernels with different length-scales.
      </Prose>

      <H3>9.4 Non-Gaussian likelihoods</H3>

      <Prose>
        The exact GP posterior is only tractable when the likelihood is Gaussian — i.e., for regression with homoscedastic Gaussian noise. For binary classification (Bernoulli likelihood with logistic or probit link), count data (Poisson), or ordinal outcomes, the posterior is non-Gaussian and must be approximated. The three main approximation strategies are: (1) <strong>Laplace approximation</strong>: find the posterior mode (MAP estimate of the latent function values), approximate the posterior as a Gaussian centered there — fast but inaccurate for highly non-Gaussian posteriors; (2) <strong>Expectation Propagation (EP)</strong>: approximate each likelihood factor independently with a Gaussian — more accurate than Laplace, used in GPML software; (3) <strong>Variational inference (VI)</strong>: optimize a tractable variational lower bound — used in GPflow's SVGP and easily combined with sparse approximations. Sklearn's <Code>GaussianProcessClassifier</Code> uses the Laplace approximation.
      </Prose>

      <H3>9.5 Misspecified priors and model checking</H3>

      <Prose>
        GPs are fully probabilistic and produce calibrated uncertainty — but only if the model is correctly specified. A GP with a prior that is misspecified (wrong kernel, wrong hyperparameter range, wrong noise model) can produce confidently wrong predictions. Calibration checks are essential: for regression, verify that roughly 95% of held-out test points fall within the 95% prediction interval. Systematic over-coverage (CI too wide) suggests over-regularized hyperparameters; systematic under-coverage suggests the kernel fails to capture the true correlation structure. Use posterior predictive checks — compare the distribution of residuals <Code>{"(y - μ_*) / σ_*"}</Code> to a standard normal; deviations indicate model misspecification.
      </Prose>

      <H3>9.6 Heteroscedastic noise</H3>

      <Prose>
        Standard GPs assume homoscedastic noise: the same noise variance <Code>{"σ_n^2"}</Code> everywhere. If the true noise varies with the input (e.g., measurement uncertainty increases away from a sensor), the homoscedastic GP will over-smooth in low-noise regions and produce overconfident predictions. The fix is a heteroscedastic GP: model the log noise variance as another GP (Goldberg et al., 1998; Kersting et al., 2007). This doubles the number of GPs to fit and is significantly more complex, but it is available in GPyTorch and GPflow. For mild heteroscedasticity, the <Code>WhiteKernel</Code> in sklearn captures a constant noise floor — not a full solution but often good enough.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were WebSearch-verified for author, year, venue, and main claims. Read them in this order to understand the full intellectual lineage of Gaussian processes.
      </Prose>

      <StepTrace
        label="primary literature — verified citations"
        steps={[
          {
            label: "Krige 1951 — Statistical interpolation for gold ore estimation",
            render: () => (
              <Prose>
                Krige, D.G. (1951). "A Statistical Approach to Some Basic Mine Valuation Problems on the Witwatersrand." <em>Journal of the Chemical, Metallurgical and Mining Society of South Africa</em>, 52(6):119–139. Based on Krige's M.Sc. thesis at the University of the Witwatersrand. The first formalization of distance-weighted spatial interpolation: ore grade at an unsampled location is estimated by a weighted average of nearby drill-hole measurements, where weights depend on spatial correlation. Krige derived these weights empirically by fitting variograms to mine data. The method was named "kriging" by Matheron in his honor. This paper marks the beginning of what would become the Gaussian process literature, though that connection was not made explicit until O'Hagan (1978).
              </Prose>
            ),
          },
          {
            label: "Matheron 1963 — Principles of Geostatistics",
            render: () => (
              <Prose>
                Matheron, G. (1963). "Principles of Geostatistics." <em>Economic Geology</em>, 58(8):1246–1266. DOI: 10.2113/gsecongeo.58.8.1246. Matheron placed Krige's empirical method on a rigorous mathematical foundation using second-order stationary random function theory. He introduced the semivariogram as the fundamental tool for characterizing spatial correlation and derived the kriging equations — the same linear system that GP regression produces. He coined the term "kriging" explicitly in this paper. The mathematical framework Matheron developed is identical, up to parameterization, to the GP framework used in machine learning today. Geostatisticians and ML researchers were solving the same problem independently for 35 years before the connection was widely recognized.
              </Prose>
            ),
          },
          {
            label: "Williams & Rasmussen 1996 — Gaussian Processes for Regression (NeurIPS)",
            render: () => (
              <Prose>
                Williams, C.K.I. and Rasmussen, C.E. (1996). "Gaussian Processes for Regression." <em>Advances in Neural Information Processing Systems 8</em>. MIT Press. Proceedings at proceedings.neurips.cc/paper/1995. The paper that introduced GPs to the machine learning community as a principled competitor to neural networks for nonparametric regression. Williams and Rasmussen showed that (1) the GP posterior is computable in closed form; (2) kernel hyperparameters can be optimized via marginal likelihood rather than cross-validation; (3) GPs outperform neural networks on several benchmark problems when data is scarce. This paper crystallized the connection between Bayesian nonparametric regression and the geostatistical literature, and sparked the decade of GP research that followed.
              </Prose>
            ),
          },
          {
            label: "Rasmussen & Williams 2006 — GPML book (the definitive reference)",
            render: () => (
              <Prose>
                Rasmussen, C.E. and Williams, C.K.I. (2006). <em>Gaussian Processes for Machine Learning</em>. MIT Press. ISBN: 978-0-262-18253-9. Available free at gaussianprocess.org/gpml. The comprehensive treatise on GP theory and practice. Covers: GP regression and classification in complete mathematical detail; kernel design and composition; sparse approximations; connections to SVM, neural networks, and splines; model selection; and extensions to multi-output GPs. Chapter 2 (regression) and Chapter 5 (model selection and adaptation) are the most practically important. The notation used in virtually all subsequent GP papers follows this book. If you read one GP reference, read this.
              </Prose>
            ),
          },
          {
            label: "Snoek, Larochelle & Adams 2012 — Practical Bayesian Optimization (NeurIPS)",
            render: () => (
              <Prose>
                Snoek, J., Larochelle, H. and Adams, R.P. (2012). "Practical Bayesian Optimization of Machine Learning Algorithms." <em>Advances in Neural Information Processing Systems 25</em>, pp. 2960–2968. arXiv:1206.2944. The paper that made Bayesian optimization mainstream in ML. Showed that GP-based BO with expected improvement acquisition could tune neural network hyperparameters to match or exceed human expert performance. Introduced parallel BO via expected improvement under a Kriging believer fantasy model, enabling asynchronous batch evaluations. Spawned the Spearmint software and directly influenced subsequent tools including BoTorch and the GP components of Optuna. The mathematical framework in this paper is a direct application of the Rasmussen-Williams GP regression machinery to the problem of optimizing expensive black-box functions.
              </Prose>
            ),
          },
          {
            label: "Hensman, Fusi & Lawrence 2013 — Gaussian Processes for Big Data (UAI)",
            render: () => (
              <Prose>
                Hensman, J., Fusi, N. and Lawrence, N.D. (2013). "Gaussian Processes for Big Data." <em>Proceedings of the 29th Conference on Uncertainty in Artificial Intelligence (UAI 2013)</em>. arXiv:1309.6835. Available at auai.org/uai2013/prints/papers/244.pdf. The paper that cracked the O(n³) bottleneck for GP regression. By framing sparse GP approximation as variational inference over inducing variables, Hensman et al. obtained a stochastic lower bound on the marginal likelihood that can be optimized with mini-batch SGD. Training cost drops to O(nm² + m³) per epoch. The SVGP framework is directly implemented in GPflow and is the standard approach for GP regression with n {">"} 10k. Extended immediately to non-Gaussian likelihoods by adding the appropriate likelihood term to the ELBO.
              </Prose>
            ),
          },
          {
            label: "Wilson & Nickisch 2015 — KISS-GP (ICML)",
            render: () => (
              <Prose>
                Wilson, A.G. and Nickisch, H. (2015). "Kernel Interpolation for Scalable Structured Gaussian Processes (KISS-GP)." <em>Proceedings of the 32nd International Conference on Machine Learning (ICML 2015)</em>. arXiv:1503.01057. Proceedings at proceedings.mlr.press/v37/wilson15.html. Introduced the Structured Kernel Interpolation (SKI) framework, which places inducing points on a regular grid and uses local cubic interpolation to approximate kernel evaluations. The resulting kernel matrix has Kronecker/Toeplitz structure, enabling O(n + m log m) matrix-vector products via FFT. Combined with iterative conjugate gradient solvers (instead of Cholesky), this reduces inference to O(n) per CG iteration. KISS-GP is the architectural foundation of GPyTorch's scalability and enables exact-approximate GP inference on datasets with millions of points in 1D.
              </Prose>
            ),
          },
          {
            label: "Rahimi & Recht 2007 — Random Features for Kernel Machines (NeurIPS)",
            render: () => (
              <Prose>
                Rahimi, A. and Recht, B. (2007). "Random Features for Large-Scale Kernel Machines." <em>Advances in Neural Information Processing Systems 20</em>, pp. 1177–1184. Proceedings at proceedings.neurips.cc/paper/2007. Proved that any shift-invariant kernel (including RBF) is the Fourier transform of a probability distribution over frequencies. This means kernel evaluations can be approximated by inner products of low-dimensional random feature maps: sample D frequencies from the spectral distribution, map inputs to 2D-dimensional features via sine/cosine functions, and run linear regression. Training costs drop from O(n³) to O(nD). The approximation error is O(1/sqrt(D)) uniformly over all inputs. This insight connected kernel methods to linear methods and enabled GP-like modeling at n in the millions, at the cost of some approximation error and the loss of exact posterior inference.
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
        Work through these before moving to the next topic. Answers are below each exercise — resist the urge to read ahead.
      </Prose>

      <H3>Exercise 1 (Recall)</H3>
      <Prose>
        Write the two GP posterior equations. What do <Code>{"K_{*X}"}</Code>, <Code>{"K_{XX}"}</Code>, and <Code>{"K_{**}"}</Code> represent? Why does the posterior variance formula always subtract a positive semidefinite term from the prior variance?
      </Prose>
      <Callout type="answer" title="Answer 1">
        {"The GP posterior mean is: μ_* = K_{*X} (K_{XX} + σ_n^2 I)^{-1} y. The posterior covariance is: Σ_* = K_{**} - K_{*X} (K_{XX} + σ_n^2 I)^{-1} K_{X*}."}
        {" K_{XX} is the n×n covariance matrix among training inputs: [K_{XX}]_{ij} = k(x_i, x_j). K_{*X} is the n_*×n matrix of kernel values between test and training inputs. K_{**} is the n_*×n_* covariance matrix among test inputs alone."}
        {" The subtracted term K_{*X}(K_{XX}+σ_n^2 I)^{-1}K_{X*} is always positive semidefinite (it is a quadratic form with a positive definite center matrix). Subtracting it from the prior covariance K_{**} gives the posterior covariance — which is always ≤ the prior. This is the mathematical statement that observing data can only reduce uncertainty, never increase it. At a test point far from all training inputs, K_{*X} ≈ 0, the subtracted term is negligible, and the posterior ≈ prior. Near a training point, K_{*X} is large and the posterior variance is nearly zero (for noise-free data)."}
      </Callout>

      <H3>Exercise 2 (Derivation)</H3>
      <Prose>
        Interpret the three terms of the GP marginal log-likelihood: <Code>{"-½ yᵀ(K+σ²I)⁻¹y - ½ log|K+σ²I| - n/2 log(2π)"}</Code>. Why does maximizing this jointly learn both the data fit and the model complexity? How is this different from choosing hyperparameters by cross-validation?
      </Prose>
      <Callout type="answer" title="Answer 2">
        {"The three terms are: (1) Data fit: -½ yᵀ(K+σ²I)^{-1}y — this is the squared Mahalanobis distance of y from zero under the model. Large when the model's predicted covariance structure does not match the observed data pattern. Maximizing this pushes hyperparameters to explain the data. (2) Complexity penalty: -½ log|K+σ²I| — the log-determinant of the covariance matrix. More flexible kernels (shorter length-scale, larger signal variance) have larger determinants → larger penalties. This term penalizes models that can explain any data pattern. (3) Normalizing constant: -n/2 log(2π) — does not depend on hyperparameters, irrelevant for optimization."}
        {" Together, terms (1) and (2) implement automatic Occam's razor: a model that fits the data with less complexity (higher log-det penalty but better data fit) is preferred over one that overfits. This is qualitatively different from cross-validation: CV estimates generalization by holding out data and is expensive (K×n_configs training runs). Marginal likelihood is computed from training data alone in one forward pass — it is faster, analytically grounded, and does not waste data on validation."}
      </Callout>

      <H3>Exercise 3 (Applied)</H3>
      <Prose>
        You fit an RBF GP to a 1D dataset and notice that the 95% confidence intervals contain only 40% of held-out test points. List two distinct root causes and the appropriate fix for each.
      </Prose>
      <Callout type="answer" title="Answer 3">
        {"Poor calibration (CI covers only 40% when it should cover 95%) means the model is overconfident — the posterior variance is too small."}
        {" Cause 1: Length-scale too long (over-smoothing). When l is too large, the posterior variance collapses because the kernel says all points are highly correlated — the model thinks it knows the function everywhere. Fix: check that l was optimized with multiple restarts. Run the optimizer from 10 different initializations covering 3-4 orders of magnitude and take the best marginal likelihood. If the training data shows rapid variation, add a Matern-3/2 or Matern-5/2 kernel instead of RBF."}
        {" Cause 2: Noise variance too small. If σ_n is underestimated (perhaps due to local optima or a near-noise-free initialization), the model attributes all training residuals to function variation rather than noise, and the posterior is overconfident. Fix: inspect the optimized σ_n against the empirical standard deviation of residuals at training inputs. If σ_n << std(y - μ_train), the noise is misspecified. Add a WhiteKernel to the kernel sum with a broad prior, or initialize log_sn with log(std(y)/2) rather than log(0.1)."}
      </Callout>

      <H3>Exercise 4 (Conceptual)</H3>
      <Prose>
        A colleague says: "GP regression is just kernel ridge regression with a Bayesian interpretation." Are they correct? What does KRR give you that GP doesn't, and what does GP give you that KRR doesn't?
      </Prose>
      <Callout type="answer" title="Answer 4">
        {"Correct, with an important caveat. The GP posterior mean is mathematically identical to the KRR prediction: both are α = (K_XX + λI)^{-1} y followed by k_* · α where λ = σ_n^2 in the GP. So the predictions (mean only) are the same."}
        {" What KRR gives you that GP doesn't: nothing extra for predictions. KRR is simply the point estimate."}
        {" What GP gives you that KRR doesn't: (1) the full posterior — not just the mean but also the posterior variance (uncertainty) at each test point, which KRR has no principled analog for; (2) a principled method for selecting the regularization parameter λ via marginal likelihood rather than cross-validation; (3) a generative probabilistic model that enables sampling from the posterior distribution, computing expected improvement for Bayesian optimization, and making probabilistic statements about future observations; (4) extension to non-Gaussian likelihoods via approximate inference. The GP is strictly more informative than KRR — it provides everything KRR does, plus principled uncertainty quantification."}
      </Callout>

      <H3>Exercise 5 (Debugging)</H3>
      <Prose>
        You are running GP Bayesian optimization on a function with 8 input dimensions. After 50 trials, the GP surrogate predicts near-zero variance everywhere — the acquisition function is nearly constant and all new points cluster in a tiny region. What is wrong and how do you fix it?
      </Prose>
      <Callout type="answer" title="Answer 5">
        {"The symptom (near-zero variance everywhere) indicates the RBF kernel's length-scale has been optimized to a very small value (over-fitting to noise) or — more likely in 8 dimensions — the isotropic length-scale has collapsed due to the curse of dimensionality."}
        {" Root cause: In 8 dimensions, all pairwise squared distances ||x - x'||^2 concentrate around 8 × 2σ^2 (where σ^2 is the per-dimension variance). If the length-scale l is optimized over a bounded range, it may converge to a value where k(x, x') ≈ 0 for all pairs of distinct points — the kernel sees all points as independent. A marginal likelihood optimizer can then get stuck here because the gradient is near zero."}
        {" Fixes: (1) Switch to ARD (Automatic Relevance Determination): use one length-scale per dimension. With ARD, the optimizer can discover which dimensions matter and set long l for uninformative dimensions and short l for informative ones. In sklearn: use RBF(length_scale=[1.0]*8) instead of RBF(length_scale=1.0). In GPyTorch: use gpytorch.kernels.RBFKernel(ard_num_dims=8). (2) Add bounded priors on the length-scale: constrain l to [0.1, 10.0] in the optimizer to prevent collapse. (3) Normalize the input space to [0,1]^d before fitting the GP. (4) If 8 dimensions is genuinely high for the number of trials (50), consider switching to a tree-based surrogate (SMAC, TPE) that does not assume a kernel smoothness structure."}
      </Callout>

      <H3>Exercise 6 (Synthesis)</H3>
      <Prose>
        You want to use GP regression on a dataset of n = 50,000 examples with d = 5 input features. Describe the exact approach you would take: which library, which kernel, which approximation (if any), how many inducing points, and how you would validate the model.
      </Prose>
      <Callout type="answer" title="Answer 6">
        {"50k examples with d=5 is beyond exact GP (exact Cholesky requires ~3GB RAM for K_XX). The right approach:"}
        {" Library: GPyTorch with GPflow as an alternative. Both support SVGP and GPU acceleration."}
        {" Kernel: Matern-5/2 with ARD (5 separate length-scales for 5 dimensions). Matern-5/2 is twice-differentiable and less unrealistically smooth than RBF. ARD prevents length-scale collapse in d=5. Add a constant mean function if the data has a non-zero mean."}
        {" Approximation: SVGP (Hensman et al., 2013) with m=500 inducing points. Place inducing points initially on a k-means++ grid of the training data. Training cost is O(n × m^2) per epoch = O(50000 × 250000) = O(1.25 × 10^10) — feasible with GPU and mini-batches of 256."}
        {" Optimization: Use Adam with learning rate 0.01 on the ELBO. Train for 50-100 epochs, monitoring ELBO convergence. Use a separate validation set (10% held out) to check for overfitting."}
        {" Validation: (1) Calibration check: compute the fraction of validation points within the 95% prediction interval — should be ~0.95. (2) Standardized residuals: (y - μ_*) / σ_* should be approximately N(0,1) — check with a Q-Q plot. (3) Compare the marginal likelihood (or ELBO) across different choices of m (100, 200, 500) to assess the inducing point approximation quality. (4) For regression: compute RMSE and negative log-predictive-density (NLPD) on the held-out set — NLPD penalizes both inaccurate means and miscalibrated variances."}
      </Callout>

    </div>
  ),
};

export default gpContent;
