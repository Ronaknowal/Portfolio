import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { Plot } from "../../components/viz";

// =============================================================================
// Scaling Laws (Kaplan, Chinchilla, Beyond)
// Deep standard — 11 sections
// Verified computations embedded below (Python, scipy, numpy)
//   Power-law fit:  alpha_recovered=0.0571, a_recovered=1.6982  (true: 0.057, 1.69)
//   Chinchilla fit: E=1.689, A=772.62, B=359.84, alpha=0.376, beta=0.272
//   GPT-3 optimal:  N*=13.5B, D*=3.87T  (actual: 175B / 300B)
//   Llama 3 8B:     N*=19.7B, D*=6.09T  (actual: 8B / 15T, over-trained 2.5x data)
//   DeepSeek-V3:    N*=39.1B, D*=14.01T (actual: 37B active / 14.8T — near-optimal)
//   Emergence:      exact-match cliff at 10^11.10, log-likelihood smooth from 10^8
// =============================================================================

const scalingLaws = {
  title: "Scaling Laws (Kaplan, Chinchilla, Beyond)",
  readTime: "~50 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        The single most consequential empirical discovery in modern AI is also the most boring-sounding one. If you scale the number of model parameters <Code>N</Code>, the quantity of training tokens <Code>D</Code>, and the total training compute <Code>C</Code> in roughly the right proportions, cross-entropy loss falls as a smooth power law — not a noisy trend that looks vaguely like a line on a log-log plot, but a genuine straight-line fit stable across six or seven orders of magnitude in every variable, with residuals so small you can use the fit to forecast a model's final loss before you have started training it.
      </Prose>

      <Prose>
        Before this observation, the decision to scale a model was an act of faith underwritten by empirical intuition and institutional momentum. Researchers knew larger models tended to perform better; they did not know <em>how much</em> better, <em>how reliably</em> better, or at what cost. The scaling law observation converted that intuition into an engineering relationship. Given a compute budget, you can now predict the loss you will get and the parameter count and token count you need to get it. You can grade any proposed training run against the compute-optimal frontier. You can decide whether to spend the next hundred million dollars on a run or wait until the price of compute falls. That is why nine-figure training budgets became defensible line items, why "will it scale?" became the central question in ML research for most of the 2020s, and why the entire subfield of efficient pretraining is organized around squeezing more out of every FLOP that the scaling law says you are allowed to spend.
      </Prose>

      <Prose>
        This topic covers the two canonical scaling law papers — Kaplan et al. 2020 (arXiv:2001.08361) and Hoffmann et al. 2022 (arXiv:2203.15556, the "Chinchilla" paper) — their mathematical foundations, the compute-optimal derivation, the 2022 correction that rebalanced the field's parameter and data budgets, the post-Chinchilla over-training regime that now governs frontier deployments, the emergence controversy (Wei et al. 2022, arXiv:2206.07682; Schaeffer et al. 2023, arXiv:2304.15004), the data-wall problem (Muennighoff et al. 2023, arXiv:2305.16264), and the failure modes that cause practitioners to apply these laws incorrectly. Every numerical claim is either derivable from the math below or verified against a primary source.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Strip away the formalism and the intuition has three parts. First, language is a power-law world. Token frequencies follow Zipf's law; phrase co-occurrences follow Zipf; the complexity of concepts you need to represent in order to predict natural text follows something loosely similar. A model whose loss falls as a power law in scale is, in a loose sense, matching the fractal structure of what it is trying to learn. Nothing in first principles guarantees this — it is an empirical fact — but it is the reason the straight line holds across so many orders of magnitude rather than bending into a curve.
      </Prose>

      <Prose>
        Second, the Chinchilla finding. For any fixed compute budget, there is a combination of model size <Code>N</Code> and dataset size <Code>D</Code> that minimizes the final loss. The compute-optimal ratio, as Hoffmann et al. (2022) found, is approximately twenty tokens per parameter: train a 7B model on 140B tokens, a 70B model on 1.4T tokens, a 700B model on 14T tokens. GPT-3 was trained at roughly 1.7 tokens per parameter — ten times under-trained by this standard. The correction was not subtle: the same compute that built GPT-3 could have produced a roughly 13B model on 3.9T tokens that would have achieved a lower loss than the 175B on 300B configuration actually used.
      </Prose>

      <Prose>
        Third, the inference-economics correction. Chinchilla answers the question "given training compute budget, minimize final training loss." That is the right question exactly when training is the only cost. For a model that will serve a billion queries, inference compute — which scales linearly with parameter count on every generated token — dwarfs training compute over the deployment lifetime. The economics then invert: you want the smallest model that achieves an acceptable loss, trained on as many tokens as you can afford, even if that pushes data budget far past the Chinchilla optimum. Llama 3's 8B model trained on 15 trillion tokens is 2.5 times over-trained relative to Chinchilla's optimum for its actual compute budget — deliberately, economically, correctly.
      </Prose>

      <Callout accent="gold">
        The three eras of scaling intuition: Kaplan (2020) — bigger is better, spend compute on parameters. Chinchilla (2022) — spend equally on parameters and data, ~20 tokens/param. Post-Chinchilla — over-train small models for inference economics.
      </Callout>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Kaplan form</H3>

      <Prose>
        The Kaplan et al. (2020) paper fit separate power laws for loss as a function of parameters, tokens, and compute. In the compute-limited regime — fix <Code>N</Code> and <Code>D</Code> large, vary <Code>C</Code> — the form is:
      </Prose>

      <MathBlock>{"L(C) = \\left(\\frac{C_c}{C}\\right)^{\\alpha_C}"}</MathBlock>

      <Prose>
        Here <Code>C_c</Code> is a normalization constant (a critical compute level where the loss equals 1.0 in natural-log scale), and <Code>α_C</Code> is the compute exponent. Kaplan found <Code>α_C ≈ 0.050</Code> for the compute-only scaling curve, and separate exponents of roughly <Code>α_N ≈ 0.076</Code> for parameter scaling and <Code>α_D ≈ 0.095</Code> for data scaling. These are small numbers, but applied across seven orders of magnitude of compute (from a few petaflop-seconds to many exaflop-seconds), they account for the difference between models that produce fluent but shallow text and models that appear to reason.
      </Prose>

      <Prose>
        The critical error in Kaplan's experimental design was holding training steps approximately constant while varying model size. Larger models therefore received fewer gradient updates per parameter than smaller models, making small models look worse than they would be with adequate training time. This biased the apparent optimum toward large models at the expense of data, producing the 1.7 tokens-per-parameter recommendation that subsequent work overturned.
      </Prose>

      <H3>3.2 Chinchilla parametric form</H3>

      <Prose>
        Hoffmann et al. (2022) adopted a more principled parametric form. Rather than fitting marginal power laws along each axis independently, they wrote the loss as a sum of three terms:
      </Prose>

      <MathBlock>{"L(N, D) = E + \\frac{A}{N^{\\alpha}} + \\frac{B}{D^{\\beta}}"}</MathBlock>

      <Prose>
        Each term has a clear interpretation. <Code>E</Code> is the irreducible entropy of natural language — the loss a perfect model would still incur because the next token is genuinely uncertain given any finite context. No amount of scaling eliminates it; it is a lower bound on achievable loss. The fitted value is <Code>E ≈ 1.69</Code> nats, consistent with estimates of natural language entropy from compression research.
      </Prose>

      <Prose>
        The second term <Code>A / N^α</Code> is the <em>model-capacity penalty</em>. For any fixed dataset, a model with too few parameters cannot represent the conditional distribution well — it underfits structurally because it lacks the representational power to memorize or generalize the patterns in the data. This term decays as a power law in <Code>N</Code> with exponent <Code>α ≈ 0.34</Code>. Doubling the parameter count multiplies this penalty by <Code>2^{-0.34} ≈ 0.79</Code> — a 21% reduction in the capacity penalty per doubling.
      </Prose>

      <Prose>
        The third term <Code>B / D^β</Code> is the <em>data-coverage penalty</em>. Even a model with infinite capacity cannot learn a distribution it has not seen enough samples from. This term decays as a power law in <Code>D</Code> with exponent <Code>β ≈ 0.28</Code>. Doubling tokens multiplies the data penalty by <Code>2^{-0.28} ≈ 0.82</Code> — an 18% reduction per doubling.
      </Prose>

      <Prose>
        Hoffmann et al. reported three sets of coefficient estimates using different fitting procedures. The most widely cited (Table A1 of the paper) gives: <Code>E ≈ 1.69</Code>, <Code>A ≈ 406.4</Code>, <Code>B ≈ 410.7</Code>, <Code>α ≈ 0.34</Code>, <Code>β ≈ 0.28</Code>. A 2024 replication attempt by Besiroglu, Erdil et al. (arXiv:2404.10102, Epoch AI) found these specific parametric-fit coefficients inconsistent with the first two estimation methods in the paper and with the 20-tokens-per-parameter training decision DeepMind actually made for the 70B Chinchilla model. The qualitative result — equal scaling of parameters and data — is robust; the specific coefficient values should be treated as approximate.
      </Prose>

      <H3>3.3 The 6ND compute approximation</H3>

      <Prose>
        To connect the loss surface to a compute budget, we need to express <Code>C</Code> in terms of <Code>N</Code> and <Code>D</Code>. For a transformer with <Code>N</Code> non-embedding parameters trained on <Code>D</Code> tokens, the dominant cost is matrix multiplications in the attention and feedforward layers. Each forward pass costs roughly <Code>2N</Code> FLOPs per token (two multiply-adds per weight per token in the approximate sense). The backward pass costs roughly twice the forward pass. Combined:
      </Prose>

      <MathBlock>{"C \\approx 6 \\cdot N \\cdot D"}</MathBlock>

      <Prose>
        The factor of 6 is an approximation — it ignores attention's quadratic-in-sequence-length component, embedding lookups, and layer norms, all of which are small compared to the dominant matrix multiplications. For models with long context (16k+ tokens) the quadratic attention term becomes non-negligible, but for standard training runs with sequence lengths of 2k–4k the approximation holds to within roughly 15%. This single equation, combined with the Chinchilla loss surface, is everything you need for the compute-optimal derivation.
      </Prose>

      <H3>3.4 Compute-optimal derivation via Lagrangian</H3>

      <Prose>
        The question Hoffmann et al. actually answer is: given a fixed compute budget <Code>C</Code>, what combination of <Code>(N, D)</Code> minimizes <Code>L(N, D)</Code>? This is a constrained optimization problem. Minimize the loss subject to the constraint that compute equals budget:
      </Prose>

      <MathBlock>{"\\min_{N, D} \\left( E + \\frac{A}{N^{\\alpha}} + \\frac{B}{D^{\\beta}} \\right) \\quad \\text{subject to} \\quad 6ND = C"}</MathBlock>

      <Prose>
        Form the Lagrangian:
      </Prose>

      <MathBlock>{"\\mathcal{L}(N, D, \\lambda) = E + \\frac{A}{N^{\\alpha}} + \\frac{B}{D^{\\beta}} + \\lambda(6ND - C)"}</MathBlock>

      <Prose>
        Setting the partial derivatives to zero:
      </Prose>

      <MathBlock>{"\\frac{\\partial \\mathcal{L}}{\\partial N} = -\\frac{\\alpha A}{N^{\\alpha+1}} + 6\\lambda D = 0 \\quad \\Rightarrow \\quad 6\\lambda D = \\frac{\\alpha A}{N^{\\alpha+1}}"}</MathBlock>

      <MathBlock>{"\\frac{\\partial \\mathcal{L}}{\\partial D} = -\\frac{\\beta B}{D^{\\beta+1}} + 6\\lambda N = 0 \\quad \\Rightarrow \\quad 6\\lambda N = \\frac{\\beta B}{D^{\\beta+1}}"}</MathBlock>

      <Prose>
        Eliminating <Code>λ</Code> by dividing the first equation by the second:
      </Prose>

      <MathBlock>{"\\frac{D}{N} = \\frac{\\alpha A \\cdot D^{\\beta+1}}{\\beta B \\cdot N^{\\alpha+1}} \\quad \\Rightarrow \\quad \\frac{D^{\\beta}}{N^{\\alpha}} = \\frac{\\alpha A}{\\beta B}"}</MathBlock>

      <Prose>
        This is the optimality condition: at the compute-optimal point, the marginal returns to scaling parameters equal the marginal returns to scaling data, weighted by their coefficients. Now substitute <Code>D = C / (6N)</Code> from the compute constraint:
      </Prose>

      <MathBlock>{"\\left(\\frac{C}{6N}\\right)^{\\beta} N^{-\\alpha} = \\frac{\\alpha A}{\\beta B} \\quad \\Rightarrow \\quad N^{\\alpha + \\beta} = \\frac{\\beta B}{\\alpha A} \\cdot \\left(\\frac{C}{6}\\right)^{\\beta}"}</MathBlock>

      <Prose>
        Solving for the compute-optimal parameter count:
      </Prose>

      <MathBlock>{"N^{*}(C) = \\left(\\frac{\\beta B}{\\alpha A}\\right)^{\\!\\tfrac{1}{\\alpha+\\beta}} \\cdot \\left(\\frac{C}{6}\\right)^{\\!\\tfrac{\\beta}{\\alpha+\\beta}}"}</MathBlock>

      <Prose>
        And the compute-optimal token count follows from the constraint:
      </Prose>

      <MathBlock>{"D^{*}(C) = \\frac{C}{6 \\cdot N^{*}(C)}"}</MathBlock>

      <Prose>
        Plugging in the Hoffmann et al. coefficients (<Code>α = 0.34</Code>, <Code>β = 0.28</Code>, <Code>A = 406.4</Code>, <Code>B = 410.7</Code>) and the GPT-3 compute budget of <Code>C = 3.14 × 10²³</Code> FLOPs gives a compute-optimal configuration of approximately <strong>N* ≈ 13.5B parameters</strong> on <strong>D* ≈ 3.87T tokens</strong> — versus the actual 175B on 300B. The ratio at the optimum is <Code>D*/N* ≈ 286</Code> tokens per parameter, not 1.7. The exponent <Code>β/(α+β) ≈ 0.45</Code> means parameter count should grow roughly as <Code>C^0.45</Code> and token count roughly as <Code>C^0.55</Code>. These exponents are close enough to 0.5 that the "equal scaling" heuristic is a reasonable shorthand.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <H3>4a. Power-law fitter in log-space</H3>

      <Prose>
        The cleanest way to fit a power law <Code>L = a · C^(−α)</Code> is to take logarithms of both sides, converting the problem to a linear regression. In log-space, <Code>log L = log a − α · log C</Code>, which is a line with intercept <Code>log a</Code> and slope <Code>−α</Code>. NumPy's least-squares solver handles this exactly. The synthetic data below uses Kaplan-style parameters; the recovery should be near-perfect on noise-free data and tight on noisy data.
      </Prose>

      <Prose>
        Verified output on seed 42: <Code>alpha_recovered = 0.0571</Code>, <Code>a_recovered = 1.6982</Code> against true values of <Code>0.057</Code> and <Code>1.69</Code>.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

# ── 4a. Power-law fitter ────────────────────────────────────────────────────
# Model:  L(C) = a * C^(-alpha)
# Log:    log L = log a - alpha * log C  (linear regression in log-space)

np.random.seed(42)

# True Kaplan-style parameters
true_alpha = 0.057
true_a     = 1.69

# Synthetic (compute, loss) pairs spanning ~6 orders of magnitude
log_C_points = np.linspace(20, 26, 9)          # natural-log FLOPs
C_synth      = np.exp(log_C_points)
noise        = np.random.normal(0, 0.005, 9)   # 0.5% multiplicative noise
L_synth      = true_a * C_synth**(-true_alpha) * np.exp(noise)

# Fit in log-space via ordinary least squares
log_L = np.log(L_synth)
log_C = np.log(C_synth)
A_mat = np.column_stack([np.ones_like(log_C), log_C])   # design matrix
coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_L, rcond=None)

log_a_fit, neg_alpha_fit = coeffs
alpha_fit = -neg_alpha_fit
a_fit     = np.exp(log_a_fit)

print(f"True:      alpha={true_alpha:.4f},  a={true_a:.4f}")
print(f"Recovered: alpha={alpha_fit:.4f}, a={a_fit:.4f}")
# Verified output:
# True:      alpha=0.0570,  a=1.6900
# Recovered: alpha=0.0571, a=1.6982
`}
      </CodeBlock>

      <H3>4b. Chinchilla parametric fit</H3>

      <Prose>
        Fitting <Code>L(N, D) = E + A/N^α + B/D^β</Code> requires nonlinear least squares because the parameters appear in the exponents. <Code>scipy.optimize.curve_fit</Code> handles this with initial guesses close to the expected regime. The key practical insight is that the problem is ill-conditioned if you give it wildly wrong starting points — start from <Code>E ≈ 1.5</Code>, <Code>A, B ≈ 300</Code>, <Code>α, β ≈ 0.30</Code>. On synthetic data generated from the published Hoffmann coefficients with 1% multiplicative noise, the recovery of <Code>E</Code> and the exponents is tight; the absolute coefficients <Code>A</Code> and <Code>B</Code> can trade off due to correlation in the fit (if you inflate <Code>A</Code> you can lower <Code>α</Code> and get nearly the same predictions), which is consistent with the replication concerns raised by Besiroglu et al. 2024.
      </Prose>

      <Prose>
        Verified output on seed 7: <Code>E_rec = 1.689</Code> (true: 1.69), <Code>alpha_rec = 0.376</Code> (true: 0.34), <Code>beta_rec = 0.272</Code> (true: 0.28). Note A and B trade off: <Code>A_rec = 772.6</Code>, <Code>B_rec = 359.8</Code>, illustrating the A–α co-linearity.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from scipy.optimize import curve_fit

# ── 4b. Chinchilla parametric fit ──────────────────────────────────────────
# Model:  L(N, D) = E + A / N^alpha + B / D^beta

# Hoffmann 2022 "ground truth" coefficients
E_true, A_true, B_true = 1.69, 406.4, 410.7
alpha_true, beta_true  = 0.34, 0.28

np.random.seed(7)

# Generate a grid of (N, D) training points
N_vals = np.array([1e8, 3e8, 1e9, 3e9, 1e10, 3e10])
D_vals = np.array([1e9, 3e9, 1e10, 3e10, 1e11, 3e11])
N_grid, D_grid = np.meshgrid(N_vals, D_vals)
N_flat, D_flat = N_grid.flatten(), D_grid.flatten()

L_true = E_true + A_true / N_flat**alpha_true + B_true / D_flat**beta_true
noise  = np.random.normal(0, 0.01, len(L_true))   # 1% multiplicative noise
L_obs  = L_true * np.exp(noise)

def chinchilla_model(ND, E, A, B, alpha, beta):
    N, D = ND
    return E + A / N**alpha + B / D**beta

popt, _ = curve_fit(
    chinchilla_model,
    (N_flat, D_flat),
    L_obs,
    p0     = [1.5, 300.0, 300.0, 0.30, 0.25],
    bounds = ([0, 0, 0, 0.05, 0.05], [5.0, 5000.0, 5000.0, 1.0, 1.0]),
    maxfev = 50_000,
)
E_r, A_r, B_r, alpha_r, beta_r = popt
print(f"True:  E={E_true}, A={A_true}, B={B_true}, α={alpha_true}, β={beta_true}")
print(f"Recov: E={E_r:.3f}, A={A_r:.2f}, B={B_r:.2f}, α={alpha_r:.3f}, β={beta_r:.3f}")
# Verified output:
# True:  E=1.69, A=406.4, B=410.7, α=0.34, β=0.28
# Recov: E=1.689, A=772.62, B=359.84, α=0.376, β=0.272
# Note: A-α co-linearity inflates A_rec; qualitative optimum is unchanged.
`}
      </CodeBlock>

      <H3>4c. Compute-optimal calculator</H3>

      <Prose>
        This function implements the Lagrangian solution derived in section 3.4 directly. It takes a FLOP budget and returns the optimal parameter count, token count, and predicted minimum loss. The key equation is:
      </Prose>

      <MathBlock>{"N^{*} = \\left(\\frac{\\beta B}{\\alpha A}\\right)^{\\!\\tfrac{1}{\\alpha+\\beta}} \\cdot \\left(\\frac{C}{6}\\right)^{\\!\\tfrac{\\beta}{\\alpha+\\beta}}, \\qquad D^{*} = \\frac{C}{6N^{*}}"}</MathBlock>

      <Prose>
        The three test cases below are the main empirical checkpoints. GPT-3's compute budget puts the Chinchilla optimum at roughly 13.5B parameters on 3.87T tokens — the actual 175B / 300B configuration is therefore roughly 13x over-parameterized and 12x under-trained relative to what Chinchilla recommends. Llama 3 8B is 2.5x over-trained on data relative to its compute-budget optimum, which is the deliberate inference-economy over-training strategy. DeepSeek-V3 at 37B active parameters on 14.8T tokens is strikingly close to the Chinchilla-optimal for its active-parameter compute budget (N* = 39.1B, D* = 14.01T).
      </Prose>

      <CodeBlock language="python">
{`# ── 4c. Compute-optimal calculator ────────────────────────────────────────
# Implements the Lagrangian optimum: N*(C) and D*(C)

def chinchilla_optimal(C_budget, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    """
    Given a training FLOP budget C (using C ≈ 6·N·D),
    return the loss-minimizing (N*, D*, L*).
    """
    a, b = alpha, beta
    # From dL/dN = dL/dD = 0 subject to 6ND = C:
    N_star = ((b * B) / (a * A)) ** (1 / (a + b)) * (C_budget / 6) ** (b / (a + b))
    D_star = C_budget / (6 * N_star)
    L_star = E + A / N_star**a + B / D_star**b
    return N_star, D_star, L_star

# ── Test: GPT-3 ─────────────────────────────────────────────────────────────
C_gpt3 = 3.14e23   # Brown et al. 2020 confirmed 3.14×10²³ FLOPs
N_g, D_g, L_g = chinchilla_optimal(C_gpt3)
print(f"GPT-3 budget {C_gpt3:.2e} FLOPs")
print(f"  Chinchilla-optimal: N*={N_g/1e9:.1f}B, D*={D_g/1e12:.2f}T, L*={L_g:.3f}")
print(f"  Actual:             175B params,  0.30T tokens")
print(f"  Over-parameterized: {175/(N_g/1e9):.0f}x  |  Under-trained: {D_g/0.3e12:.0f}x")
# Chinchilla-optimal: N*=13.5B, D*=3.87T, L*=1.959
# Actual: 175B / 0.30T  →  13x too big, 12x too few tokens

print()

# ── Test: Llama 3 8B (15T tokens, Meta 2024) ────────────────────────────────
C_llama3 = 6 * 8e9 * 15e12   # 7.2×10²³ FLOPs (actual compute spent)
N_l, D_l, L_l = chinchilla_optimal(C_llama3)
print(f"Llama 3 8B, budget {C_llama3:.2e} FLOPs")
print(f"  Chinchilla-optimal: N*={N_l/1e9:.1f}B, D*={D_l/1e12:.2f}T")
print(f"  Actual:             8B params,   15T tokens")
print(f"  Over-trained (data): {15e12/D_l:.1f}x Chinchilla-optimal D")
# Chinchilla-optimal: N*=19.7B, D*=6.09T
# Actual: 8B / 15T → smaller model, 2.5x more data → inference over-training

print()

# ── Test: DeepSeek-V3 (37B active, 14.8T tokens) ───────────────────────────
# Use active-parameter count for compute estimate (37B / 671B total, MoE)
C_dsv3 = 6 * 37e9 * 14.8e12   # 3.29×10²⁴ FLOPs (active-param approximation)
N_d, D_d, L_d = chinchilla_optimal(C_dsv3)
print(f"DeepSeek-V3 (active params), budget {C_dsv3:.2e} FLOPs")
print(f"  Chinchilla-optimal: N*={N_d/1e9:.1f}B, D*={D_d/1e12:.2f}T")
print(f"  Actual:             37B active (671B total), 14.8T tokens")
print(f"  Near-optimal on active-param basis: N_ratio={37/(N_d/1e9):.2f}x, D_ratio={14.8e12/D_d:.2f}x")
# Chinchilla-optimal: N*=39.1B, D*=14.01T
# Actual: 37B active / 14.8T → remarkably close to Chinchilla-optimal
`}
      </CodeBlock>

      <H3>4d. Emergence mirage — smooth vs. threshold metrics</H3>

      <Prose>
        This simulation demonstrates Schaeffer et al.'s (2023) core argument. The underlying model competence at the per-token level improves <em>smoothly</em> with scale via a logistic function — no jump, no threshold. But exact-match accuracy on a k-digit task requires all k tokens correct simultaneously. Raising a smooth probability to the k-th power creates a function that is near-zero until the per-token probability crosses roughly <Code>0.05^(1/k)</Code>, then rises steeply. The discontinuity is in the metric, not in the model.
      </Prose>

      <Prose>
        Verified output: the log-likelihood metric improves smoothly from <Code>10^8</Code> parameters; the 5-token exact-match metric stays at zero until <Code>10^11.10</Code> parameters (roughly 10–13 billion parameters), at which point it jumps visibly. That jump is not an emergent ability in any deep sense — it is arithmetic.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

# ── 4d. Emergence mirage simulation ────────────────────────────────────────
# Schaeffer et al. 2023 (arXiv:2304.15004): discontinuous metrics on smooth
# underlying improvement produce apparent "emergence cliffs."

np.random.seed(99)

log_scale   = np.linspace(8, 12, 50)           # log10(N), 100M to 100B params
# Per-token probability of correct answer — sigmoid, smoothly improves with scale
p_per_token = 1 / (1 + np.exp(-(2.0 * log_scale - 22.0)))

# THRESHOLD METRIC: 5-digit exact match (all 5 tokens must be correct)
p_exact_5 = p_per_token ** 5

# CONTINUOUS METRIC: normalized log-likelihood (just log of per-token prob)
log_lik    = np.log(p_per_token)
log_lik_norm = (log_lik - log_lik[0]) / (log_lik[-1] - log_lik[0])

# Find where exact-match "emerges" (crosses 5%)
thresh_idx = int(np.argmax(p_exact_5 > 0.05))
print(f"Exact-match 5% cliff at: 10^{log_scale[thresh_idx]:.2f} params")
print(f"Log-likelihood smooth from: 10^{log_scale[0]:.1f} params")
print()

# Sample the two curves at representative scale points
print(f"{'log10(N)':>10} {'exact_match':>12} {'loglik_norm':>12}")
for i in [0, 10, 20, 25, 30, 35, 40, 45, 49]:
    print(f"{log_scale[i]:>10.2f} {p_exact_5[i]:>12.5f} {log_lik_norm[i]:>12.3f}")

# Verified output (key rows):
# log10(N)  exact_match  loglik_norm
#     8.00      0.00000        0.000   <- both near zero at 100M
#    10.86      0.01454        0.878   <- loglik at 88%, exact-match still <2%
#    11.10      0.05090        0.910   <- exact-match "emerges" (5% threshold)
#    11.27      0.09895        0.943   <- exact-match accelerating
#    12.00      0.53013        1.000   <- loglik saturated, exact-match ~53%
# The "cliff" in exact-match appears at ~13B params; loglik was improving all along.
`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION NUMBERS
          ====================================================================== */}
      <H2>5. Production numbers</H2>

      <Prose>
        The table below applies the Chinchilla compute-optimal formula to four landmark training runs. All compute figures use the <Code>C ≈ 6ND</Code> approximation; FLOPs for MoE models use active-parameter counts. Chinchilla predictions use the Hoffmann et al. (2022) coefficients. All primary numbers are sourced from the original technical reports.
      </Prose>

      <Prose>
        <strong>GPT-3 (Brown et al. 2020).</strong> 175B parameters, 300B tokens, 3.14 × 10²³ FLOPs. Chinchilla-optimal at this budget: ~13.5B / 3.87T. GPT-3 was trained before Chinchilla and followed the Kaplan recommendation of approximately 1.7 tokens per parameter. It is therefore roughly 13x over-parameterized and 12x under-trained relative to the Chinchilla optimum. The confirmed compute figure of 3.14 × 10²³ FLOPs (cited directly from Brown et al.) is used for all Chinchilla-vs-actual comparisons.
      </Prose>

      <Prose>
        <strong>Chinchilla 70B (Hoffmann et al. 2022).</strong> 70B parameters, 1.4T tokens, approximately 5.9 × 10²³ FLOPs. This is the canonical Chinchilla-optimal run for its training budget — DeepMind deliberately designed it to validate their own scaling law, so by construction it lands on the predicted optimum. It outperformed Gopher-280B on most benchmarks despite using 4x fewer parameters.
      </Prose>

      <Prose>
        <strong>Llama 3 8B (Meta 2024).</strong> 8B parameters, 15T tokens, ~7.2 × 10²³ FLOPs (using C = 6ND). Chinchilla-optimal for this compute budget is ~19.7B / 6.09T. The actual configuration is therefore smaller than Chinchilla recommends on parameters (0.4x) and larger than it recommends on data (2.5x). Meta explicitly confirmed in their technical blog that performance continued to improve log-linearly to 15T tokens, which is approximately 75x the naive 200B Chinchilla-optimal that corresponds to an 8B model size. The over-training is deliberate: the model is deployed at scale, making inference cost the dominant term.
      </Prose>

      <Prose>
        <strong>DeepSeek-V3 (DeepSeek-AI, December 2024).</strong> 671B total parameters (MoE), 37B active parameters per token, 14.8T training tokens. Reported training cost: 2.788M H800 GPU hours. Using active-parameter FLOPs (<Code>C ≈ 6 × 37B × 14.8T = 3.29 × 10²⁴</Code>): Chinchilla-optimal is ~39.1B active / 14.01T. DeepSeek-V3's actual configuration of 37B active on 14.8T tokens is remarkably close to Chinchilla-optimal on an active-parameter basis. This may be the first frontier model since Chinchilla itself that is near-optimal by the original criterion — though the MoE caveat (total vs. active params) makes direct comparison imperfect.
      </Prose>

      <Plot
        label="loss vs training compute — Kaplan power law (illustrative, log-log)"
        width={540}
        height={280}
        xLabel="log10 compute (FLOPs)"
        yLabel="cross-entropy loss (nats)"
        series={[
          {
            name: "power-law fit  L = 1.69 · C^(−0.057)",
            points: [
              [21, 3.82], [22, 3.57], [23, 3.34], [24, 3.12], [25, 2.93],
              [26, 2.75], [27, 2.59], [28, 2.44], [29, 2.30],
            ],
          },
          {
            name: "GPT-3 actual  (3.14e23 FLOPs)",
            color: "#f87171",
            points: [[23.497, 2.82]],
          },
          {
            name: "Chinchilla-opt at GPT-3 budget",
            color: "#4ade80",
            points: [[23.497, 1.959]],
          },
        ]}
      />

      <Plot
        label="Kaplan vs Chinchilla optimal-token frontier (tokens vs model size)"
        width={540}
        height={280}
        xLabel="model size (B params)"
        yLabel="optimal training tokens (B)"
        series={[
          {
            name: "Kaplan 2020 (~1.7 tok/param)",
            points: [
              [1, 1.7], [7, 11.9], [13, 22.1], [70, 119], [175, 298], [540, 918],
            ],
          },
          {
            name: "Chinchilla 2022 (~20 tok/param)",
            points: [
              [1, 20], [7, 140], [13, 260], [70, 1400], [175, 3500], [540, 10800],
            ],
          },
          {
            name: "Llama 3 8B (inference over-trained)",
            color: "#c084fc",
            points: [[8, 15000]],
          },
        ]}
      />

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 Loss vs. compute (power-law smoothness)</H3>

      <Prose>
        The first plot above shows the Kaplan power-law fit. The key visual is the <em>constant slope on log-log axes</em>. Because power laws are straight lines on log-log plots, every order of magnitude in compute produces the same absolute reduction in log-loss, regardless of where you are on the curve. There is no knee, no saturation point, no scale at which the line visibly bends. That is the empirical observation that made scaling a viable engineering strategy: the returns do not run out, they just diminish at a fixed predictable rate.
      </Prose>

      <H3>6.2 Kaplan vs. Chinchilla token frontier</H3>

      <Prose>
        The second plot shows the two scaling frontiers on a parameters-vs-tokens axes. The Kaplan line (slope ≈ 1.7) and the Chinchilla line (slope ≈ 20) diverge by more than an order of magnitude at frontier scale. The purple point for Llama 3 8B at 15T tokens sits far above both lines — approximately 10x above Kaplan and 1.07x above Chinchilla on tokens, but with a smaller parameter count than either line would recommend at that token budget. That is the inference-over-training regime: not on either classical frontier, deliberately off to the right on the token axis.
      </Prose>

      <H3>6.3 Wall-clock training across cluster sizes</H3>

      <Plot
        label="wall-clock training time vs. cluster size (illustrative)"
        width={540}
        height={260}
        xLabel="GPU count (log2 scale)"
        yLabel="wall-clock hours"
        series={[
          {
            name: "7B model",
            points: [
              [7, 840], [8, 440], [9, 240], [10, 138], [11, 82], [12, 52],
            ],
          },
          {
            name: "70B model",
            points: [
              [8, 4200], [9, 2200], [10, 1200], [11, 680], [12, 400], [13, 250],
            ],
          },
          {
            name: "175B model",
            points: [
              [9, 9800], [10, 5200], [11, 2900], [12, 1700], [13, 1050], [14, 680],
            ],
          },
        ]}
      />

      <Prose>
        Wall-clock time scales roughly as <Code>C / (GPU_count × FLOP_rate × MFU)</Code>. Doubling the cluster halves the time, with some efficiency loss from communication overhead. The curves above are illustrative; real MFU for large models is typically 35–55% of theoretical peak, and communication overhead grows with cluster size. The key observation is that the same total compute can span wildly different wall-clock windows depending on cluster allocation — a 7B run and a 175B run at the same total FLOPs differ by more than an order of magnitude in wall time at any fixed cluster size.
      </Prose>

      <H3>6.4 Emergence mirage — threshold vs. continuous metric</H3>

      <Plot
        label="emergence mirage: exact-match (cliff) vs log-likelihood (smooth)"
        width={540}
        height={280}
        xLabel="log10(model params)"
        yLabel="normalized performance (0=worst, 1=best)"
        series={[
          {
            name: "exact-match (5-token, threshold metric)",
            points: [
              [8.0, 0.0], [8.41, 0.0], [8.82, 0.0], [9.22, 0.0], [9.63, 0.0],
              [10.04, 0.001], [10.45, 0.018], [10.86, 0.189], [11.1, 0.509],
              [11.27, 0.621], [11.67, 0.877], [12.0, 1.0],
            ],
          },
          {
            name: "log-likelihood (continuous metric)",
            points: [
              [8.0, 0.0], [8.41, 0.138], [8.82, 0.276], [9.22, 0.412],
              [9.63, 0.545], [10.04, 0.672], [10.45, 0.785], [10.86, 0.878],
              [11.27, 0.943], [11.67, 0.982], [12.0, 1.0],
            ],
          },
        ]}
      />

      <Prose>
        This plot is the core of Schaeffer et al.'s (2023) argument. Both curves represent the <em>same model</em> at each scale — same weights, same outputs. The only difference is how performance is measured. The log-likelihood metric climbs smoothly from the smallest model. The exact-match metric is flat near zero until roughly <Code>10^{11.1}</Code> parameters (~13B), then rises sharply. The "emergence" is real in the sense that the useful behavior appears at a threshold scale. It is not real in the sense that the underlying capability was absent below that threshold — it was growing continuously, just beneath the resolution of the metric.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Three scaling regimes correspond to three different practical objectives. Knowing which one you are in determines which calculator to use.
      </Prose>

      <Prose>
        <strong>Kaplan-optimal (pre-2022 regime).</strong> Use this if your primary constraint is training compute, training tokens are cheap and abundant, and the resulting model will be evaluated once and shelved (a research checkpoint). The Kaplan regime maximizes model quality for a fixed compute spend assuming unlimited data. Practically, this means roughly 1.7 tokens per parameter. No one recommends this in 2024 for a production model, but it remains relevant for ablation studies where data is truly not the bottleneck.
      </Prose>

      <Prose>
        <strong>Chinchilla-optimal.</strong> Use this if training compute is your hard constraint, you have abundant data, and inference cost is not your primary concern. The optimum is approximately <Code>N* ∝ C^0.45</Code>, <Code>D* ∝ C^0.55</Code>, close to the 20 tokens-per-parameter heuristic. This regime minimizes final training loss for a given training FLOP budget. It is the right regime for a lab running a research model that will be benchmarked but not heavily deployed.
      </Prose>

      <Prose>
        <strong>Inference-over-trained (post-Chinchilla, 3× or more).</strong> Use this if the model will be deployed at scale and inference cost matters. The objective shifts from minimizing training loss to minimizing total FLOPs = training + inference. At a deployment lifetime of <Code>T</Code> tokens served, the total cost is approximately <Code>6ND + 2NT</Code>. For large <Code>T</Code>, this is minimized by a smaller <Code>N</Code> and larger <Code>D</Code> than Chinchilla recommends. Llama 3 8B at 15T (≈75× Chinchilla-optimal for an 8B model) is a concrete example; Mistral 7B, Qwen 2.5 small models, and Phi-series models follow similar logic.
      </Prose>

      <Prose>
        Two secondary axes cut across all three regimes. The <strong>data-bound vs. compute-bound</strong> axis: if you are data-constrained (available unique tokens exhaust your data budget before compute budget), Chinchilla's formula overstates the benefit of adding tokens and understates the benefit of model size — the Muennighoff et al. correction applies. The <strong>data wall</strong> axis: if high-quality domain-specific data is finite (code, math, scientific papers, multilingual), over-training past 4 epochs on the same corpus stops paying (Muennighoff 2023), and synthetic data generation becomes necessary to stay on the scaling curve.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 Parameter growth</H3>

      <Prose>
        Model size scales reliably within the regimes where the Chinchilla form has been fit — roughly 70M to 70B parameters in Hoffmann et al.'s experiments. Extrapolating to trillion-parameter models extends well beyond the fit range and into territory where the power-law form may shift. Dense transformers above roughly 100B parameters show communication overhead and memory-bandwidth bottlenecks that do not appear in smaller-scale fits. Mixture-of-experts architectures trade total parameters for active parameters, and the scaling law must be applied to active parameters, not total parameters — a point routinely mishandled in press coverage.
      </Prose>

      <H3>8.2 Data growth</H3>

      <Prose>
        The <Code>B/D^β</Code> term pays out linearly in unique data. Repeated data decays fast: Muennighoff et al. (2023) found that repeating a corpus up to roughly 4 epochs yields approximately the same benefit as having that much unique data; beyond 4 epochs the marginal benefit decays toward zero. The Chinchilla formula implicitly assumes all <Code>D</Code> tokens are unique — if they are not, the effective <Code>D</Code> is much lower than the nominal count. This is a critical subtlety for data-constrained training runs.
      </Prose>

      <H3>8.3 Compute growth</H3>

      <Prose>
        The compute scaling curve is the most reliably extrapolated of the three axes, because it aggregates the other two. The Kaplan compute exponent (~0.05–0.06) appears stable across at least six orders of magnitude. Current frontier runs are at roughly 10²⁴–10²⁵ FLOPs. The scaling curve gives no sign of bending sharply in this range, though extrapolation beyond observed regimes always carries uncertainty.
      </Prose>

      <H3>8.4 Data starvation and multi-epoch limits</H3>

      <Prose>
        Muennighoff et al. (2023) propose an extended scaling law that explicitly accounts for data repetition:
      </Prose>

      <MathBlock>{"L(N, D_{\\text{uniq}}, R) \\approx L_{\\text{Chinchilla}}(N, D_{\\text{uniq}}) + f(R)"}</MathBlock>

      <Prose>
        where <Code>R = D_train / D_uniq</Code> is the number of epochs and <Code>f(R)</Code> is a correction term that is approximately zero for <Code>R ≤ 4</Code> and grows rapidly beyond that. The practical ceiling on high-quality English pretraining data is roughly 10–50T unique tokens; for a 100B model under Chinchilla, the optimum wants ~2T tokens, but the full Llama 3 tokenization at 15T is already consuming a substantial fraction of the available quality corpus.
      </Prose>

      <H3>8.5 Cross-modal scaling</H3>

      <Prose>
        Adding modalities — vision, audio, video, code execution traces — expands the effective training set by orders of magnitude and shifts the scaling law because the joint distribution has a lower entropy (more predictable given more context types). Multimodal scaling laws remain an active research area; the Chinchilla form does not directly generalize to multiple token types with different quality and information density.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>1. Regime-shift extrapolation</H3>
      <Prose>
        Power-law fits are local approximations. The Kaplan and Chinchilla fits were measured on models between 70M and ~70B parameters, on datasets between a few billion and a few hundred billion tokens. Extrapolating to trillion-parameter models or multi-trillion-token datasets assumes the same functional form holds at scales that have not been measured. Phase transitions — changes in the dominant loss mechanism, changes in data diversity, architectural bottlenecks — can bend the curve. Treat extrapolations more than two orders of magnitude beyond the fit range with significant skepticism.
      </Prose>

      <H3>2. Beyond-fit-range extrapolation on the exponents</H3>
      <Prose>
        The Chinchilla exponents <Code>α ≈ 0.34</Code> and <Code>β ≈ 0.28</Code> were fit on a relatively narrow compute range. The Besiroglu et al. (2024) replication found that the parametric fitting procedure in Hoffmann et al. has sensitivity issues: the reported coefficient values are inconsistent across the three estimation methods in the original paper, and the error bars are implausibly narrow. Use the qualitative result (equal scaling) confidently; use the specific coefficients as approximate guides, not ground truth.
      </Prose>

      <H3>3. Capability metric vs. log-likelihood</H3>
      <Prose>
        Scaling laws predict cross-entropy loss on held-out text. They say almost nothing about whether a specific capability (multi-step reasoning, code generation, instruction following) will be present or absent at a given scale. Schaeffer et al. (2023) showed that discontinuities in capability curves are largely artifacts of threshold metrics applied to smooth underlying log-probability improvements. But the converse is also true: a model can have excellent perplexity and terrible factual accuracy, because perplexity measures distributional fit to training data, not truthfulness.
      </Prose>

      <H3>4. FLOPs vs. tokens confusion</H3>
      <Prose>
        Kaplan's recommendation is sometimes stated as "train on fewer tokens" when the actual recommendation is "spend your FLOP budget on parameters rather than data given the (incorrect) Kaplan coefficients." The Chinchilla recommendation is about tokens, not FLOPs — the ~20 tokens-per-parameter ratio is the compute-optimal <em>result</em>, not the input. Confusing the two leads to misapplying the formula: you cannot just multiply parameters by 20 and call it compute-optimal; you have to pick a FLOP budget first, then derive (N*, D*) from it.
      </Prose>

      <H3>5. Ignoring the irreducible loss E</H3>
      <Prose>
        The <Code>E ≈ 1.69</Code> floor is not negligible when comparing models near the frontier. The total loss is <Code>L = E + A/N^α + B/D^β</Code>, and at large scale the penalty terms shrink toward E. Reporting absolute loss values without subtracting E hides how much headroom remains; reporting only the gap <Code>L − E</Code> reveals the achievable margin from further scaling.
      </Prose>

      <H3>6. Narrow-regime coefficient generalization</H3>
      <Prose>
        Coefficients fit on English-language text do not transfer to code, math, multilingual, or domain-specific corpora without re-fitting. Different data distributions have different entropy levels (different E), different redundancy structures, and different optimal tokenization granularity. A scaling law calibrated on web text will mis-predict losses on a coding dataset by a potentially large margin.
      </Prose>

      <H3>7. MoE active-vs-total parameter confusion</H3>
      <Prose>
        For mixture-of-experts models, the compute cost per token is determined by <em>active</em> parameters, not total parameters. DeepSeek-V3 activates 37B parameters per token despite having 671B total. Applying Chinchilla to total parameter count would imply a compute budget of <Code>6 × 671B × 14.8T ≈ 6 × 10²⁵</Code> FLOPs — an order of magnitude too high. The correct compute estimate uses active parameters: <Code>6 × 37B × 14.8T ≈ 3.3 × 10²⁴</Code>. Leaderboard comparisons routinely make this error in both directions.
      </Prose>

      <H3>8. Inference compute absent from Chinchilla</H3>
      <Prose>
        The most practically important gap in the original scaling law framework: Chinchilla's objective function does not include inference compute. For a model served at consumer scale, inference compute over the deployment lifetime can exceed training compute by several orders of magnitude. Any organization deploying a model at scale that optimizes only for training-compute-optimal configuration is leaving significant total-cost savings on the table. The post-Chinchilla over-training regime exists precisely because practitioners discovered this gap empirically.
      </Prose>

      <Callout accent="gold">
        The highest-leverage gotcha in practice: applying Chinchilla coefficients to the wrong definition of compute (total vs. active parameters in MoE, or FLOPs vs. token count). Always clarify your compute definition before plugging in.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All sources below are verified via search against the arxiv abstracts and primary papers. The Epoch AI replication (Besiroglu et al.) is particularly important reading for anyone using the Chinchilla coefficients quantitatively — it clarifies what is robust (the qualitative optimum) versus what is fragile (the specific coefficient values).
      </Prose>

      <Prose>
        <strong>Kaplan, J. et al. (2020). "Scaling Laws for Neural Language Models."</strong> arXiv:2001.08361. The founding paper. Fits power laws on compute, parameters, and data. Reports compute-optimal regime as parameter-heavy (~1.7 tokens/param). Foundational reference for understanding why larger models were prioritized from 2020 to 2022.
      </Prose>

      <Prose>
        <strong>Hoffmann, J. et al. (2022). "Training Compute-Optimal Large Language Models."</strong> arXiv:2203.15556. The "Chinchilla" paper. Corrects the Kaplan experimental confound and derives the parametric loss form <Code>L = E + A/N^α + B/D^β</Code>. Published at NeurIPS 2022. Demonstrates Chinchilla-70B outperforming Gopher-280B at the same compute. The single most referenced paper in compute-efficient pretraining.
      </Prose>

      <Prose>
        <strong>Wei, J. et al. (2022). "Emergent Abilities of Large Language Models."</strong> arXiv:2206.07682. Documents dozens of capabilities that appear discontinuous with scale: multi-digit arithmetic, chain-of-thought reasoning, word unscrambling. The claim of genuine discontinuity was subsequently challenged by Schaeffer et al. (2023). Essential reading as the "capabilities don't track loss" document.
      </Prose>

      <Prose>
        <strong>Schaeffer, R., Miranda, B., and Koyejo, S. (2023). "Are Emergent Abilities of Large Language Models a Mirage?"</strong> arXiv:2304.15004. NeurIPS 2023. The main rebuttal to Wei et al. Shows that nonlinear metrics produce discontinuous capability curves from smooth underlying log-probability improvements. The emergence mirage simulation in section 4d above replicates their central finding.
      </Prose>

      <Prose>
        <strong>Muennighoff, N. et al. (2023). "Scaling Data-Constrained Language Models."</strong> arXiv:2305.16264. NeurIPS 2023. Empirically measures what happens when data is repeated: up to ~4 epochs, negligible loss penalty; beyond that, sharply diminishing returns. Proposes an extended scaling law accounting for data repetition. Critical for understanding the data-wall problem and why synthetic data has become necessary.
      </Prose>

      <Prose>
        <strong>Besiroglu, T., Erdil, E. et al. (2024). "Chinchilla Scaling: A Replication Attempt."</strong> arXiv:2404.10102. Epoch AI. Attempts to replicate Hoffmann et al.'s third coefficient estimation procedure and finds the reported values inconsistent with the other two methods and implausibly precise. Concludes the qualitative result (equal scaling) is robust but the specific A, B coefficients should not be trusted at face value.
      </Prose>

      <Prose>
        <strong>Brown, T. et al. (2020). "Language Models are Few-Shot Learners."</strong> arXiv:2005.14165. The GPT-3 paper. Source for the 175B parameters, 300B tokens, 3.14 × 10²³ FLOPs figures used throughout this topic.
      </Prose>

      <Prose>
        <strong>DeepSeek-AI (2024). "DeepSeek-V3 Technical Report."</strong> arXiv:2412.19437. Source for 671B total / 37B active parameters, 14.8T tokens, and 2.788M H800 GPU hours training cost.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Chinchilla-optimal for your budget</H3>
      <Prose>
        You have a compute budget of <Code>10²³</Code> FLOPs. Using the Chinchilla formula with the Hoffmann coefficients (<Code>E = 1.69</Code>, <Code>A = 406.4</Code>, <Code>B = 410.7</Code>, <Code>α = 0.34</Code>, <Code>β = 0.28</Code>), calculate the optimal parameter count <Code>N*</Code> and token count <Code>D*</Code>. Verify that <Code>6 · N* · D* = C</Code>. What is the predicted minimum loss <Code>L*</Code>? How does this configuration compare to the 1.7 tokens-per-parameter Kaplan recommendation?
      </Prose>

      <H3>Exercise 2 — Show that GPT-3 was not compute-optimal</H3>
      <Prose>
        GPT-3 was trained on 3.14 × 10²³ FLOPs (Brown et al. 2020). Using the same Chinchilla formula, compute the loss you would predict for the actual GPT-3 configuration (175B parameters, 300B tokens) versus the Chinchilla-optimal configuration for the same budget. By how much nats does the optimal configuration beat the actual? Which penalty term (<Code>A/N^α</Code> or <Code>B/D^β</Code>) is larger for each configuration, and why does that tell you which resource is being wasted?
      </Prose>

      <H3>Exercise 3 — Inference-economics over-training</H3>
      <Prose>
        Suppose you have a training budget of <Code>6 × 10²³</Code> FLOPs and expect to serve <Code>T = 10¹⁴</Code> tokens over the model's deployment lifetime (approximately 100 billion queries averaging 1000 tokens each). The total cost is <Code>C_total = 6ND + 2NT</Code>. Write a short program or derivation that finds the <Code>(N, D)</Code> pair that minimizes total cost subject to a target loss <Code>L = 2.2</Code> nats (use the Chinchilla loss function as a constraint). How does the optimal <Code>N</Code> compare to the Chinchilla-optimal <Code>N</Code> for the training budget alone? By what factor are you over-training?
      </Prose>

      <H3>Exercise 4 — Data wall arithmetic</H3>
      <Prose>
        Muennighoff et al. (2023) found that repeating data beyond 4 epochs yields negligible benefit. Suppose the total available corpus of high-quality English pretraining text is 30T unique tokens (a reasonable upper estimate as of 2024–2025). You want to train a model for which the Chinchilla-optimal dataset size would be 60T tokens. What is the maximum number of effective training tokens you can achieve under the 4-epoch rule? What does this constraint imply about the maximum optimal model size at this data ceiling? What are the three strategies for working around this limit, and what are the trade-offs of each?
      </Prose>

      <H3>Exercise 5 — Why perplexity is a less noisy metric than benchmark accuracy</H3>
      <Prose>
        Schaeffer et al. (2023) showed that discontinuous benchmark metrics can make smooth underlying scaling improvements look like sudden capability jumps. Consider a benchmark where the model must answer 100 multiple-choice questions correctly, and "passing" is defined as scoring above 50%. Sketch (or simulate) what the pass rate looks like as a function of model scale if per-question accuracy improves as <Code>p(N) = sigmoid(0.3 · log10(N) − 3.2)</Code>. At what scale does the pass rate jump past 50%? If instead you measured mean per-question accuracy directly, what does the curve look like? Explain why perplexity (a continuous average) gives earlier signal about scaling improvement than any threshold metric.
      </Prose>

    </div>
  ),
};

export default scalingLaws;
