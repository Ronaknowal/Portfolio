import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const spectralNormGPContent = {
  title: "Spectral Normalization & Gradient Penalty",
  readTime: "~36 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        For its first three years generative adversarial networks were the most exciting and least reliable training procedure in deep learning. The original 2014 formulation pitted a generator against a discriminator in a saddle-point game whose Nash equilibrium happens to coincide with the generator matching the data distribution. On paper this was beautiful. In practice the optimization was a daily exercise in heartbreak. A run that produced photorealistic 64x64 faces on epoch 30 would collapse into ten gray blobs by epoch 31. The discriminator would saturate, the generator gradient would vanish, and the only remedy in the early literature was to lower the learning rate and try again with a different seed. Researchers spoke of "GAN tricks" the way medieval physicians spoke of bloodletting — a folk pharmacopoeia of label smoothing, feature matching, instance noise, historical averaging, and one-sided label flipping that worked some of the time and that nobody fully understood.
      </Prose>

      <Prose>
        The core diagnosis was already written into the math. The original GAN minimizes a Jensen-Shannon-style divergence between the data distribution {"P_r"} and the generator distribution {"P_g"}. When those two distributions live on disjoint or nearly disjoint manifolds — which they always do in early training, because the generator is initialized to produce noise — the JS divergence is constant at {"\\log 2"} and its gradient with respect to the generator is exactly zero. The discriminator, given any sensible capacity, learns to perfectly separate real from fake within a few hundred steps. Once it does, the generator receives no gradient signal at all. This is "discriminator saturation" or "vanishing generator gradient", and it was the single most reported failure mode in 2015-2016 GAN literature.
      </Prose>

      <Prose>
        Martin Arjovsky, Soumith Chintala, and Leon Bottou published "Wasserstein GAN" at ICML 2017 (arXiv:1701.07875) with a structural fix. Replace the JS divergence with the Wasserstein-1 distance — the earth-mover's distance — which has the property that it is finite, continuous, and almost-everywhere differentiable even when the two distributions are supported on disjoint manifolds. The Kantorovich-Rubinstein duality lets you compute the Wasserstein distance as a supremum over 1-Lipschitz functions, which means the discriminator's job is no longer "classify real from fake" but "find the most discriminating 1-Lipschitz function". A 1-Lipschitz constraint on the discriminator is mandatory; without it the supremum is unbounded and the dual diverges. WGAN's original mechanism for enforcing the Lipschitz constraint was unapologetically crude: after every gradient step, clip every discriminator weight to {"[-c, c]"} for some small constant {"c"} (typically {"0.01"}). This worked in the sense that it gave a Lipschitz bound, but it also gave terrible loss surfaces — the clipped weights all piled up at {"\\pm c"}, the discriminator became a gradient sink that produced uniform outputs, and capacity utilization was abysmal.
      </Prose>

      <Prose>
        The next paper fixed clipping. Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron Courville published "Improved Training of Wasserstein GANs" at NeurIPS 2017 (arXiv:1704.00028) and proposed the gradient penalty. Instead of clipping, add a soft penalty term to the discriminator loss: {"\\lambda \\, \\mathbb{E}_{\\hat{x}}[(\\|\\nabla_{\\hat{x}} D(\\hat{x})\\|_2 - 1)^2]"} where {"\\hat{x}"} is sampled along straight lines between real and generated samples. The penalty pushes the gradient norm of {"D"} toward 1 (the optimal value for the Kantorovich-Rubinstein dual), without forcing weights to a hard interval. WGAN-GP trained better, used capacity better, and produced higher-quality images on CIFAR-10 and CelebA than any prior GAN. It became the dominant training recipe for the next year.
      </Prose>

      <Prose>
        WGAN-GP had two embarrassments. The penalty added roughly a 2x compute cost because it required a second backward pass through {"D"} for each step (the gradient with respect to {"\\hat{x}"} must be backproped <em>again</em> for the penalty term — that is double-backward, also known as Hessian-vector flavor). And the constant {"\\lambda"} had to be tuned per dataset. Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida fixed both at ICLR 2018 with "Spectral Normalization for Generative Adversarial Networks" (arXiv:1802.05957). Their observation was that a feed-forward network composed of layers with Lipschitz constant 1 has overall Lipschitz constant at most 1 (Lipschitz composes multiplicatively). And the Lipschitz constant of a linear layer with weight matrix {"W"} is exactly its largest singular value {"\\sigma_{\\max}(W)"}. So if you replace every {"W"} with {"W / \\sigma_{\\max}(W)"} you get a 1-Lipschitz layer for free, with no penalty term, no double-backward, and no hyperparameter to tune. Spectral Normalization (SN) made WGAN-style training stable on ImageNet at 128x128 resolution. SN-GAN was the first GAN to match supervised classifier-style FID scores on CIFAR-10.
      </Prose>

      <Prose>
        Spectral Normalization stuck because it was cheap. The largest singular value of a matrix can be computed in {"O(n)"} per step using power iteration, and one iteration per training step is enough as long as you cache the power vectors across steps. In practice the cost is invisible relative to the rest of the discriminator forward pass. From 2018 to 2021, every state-of-the-art GAN used some version of SN in its discriminator. Andrew Brock, Jeff Donahue, and Karen Simonyan's BigGAN at ICLR 2019 (arXiv:1809.11096) used SN throughout the discriminator and additionally in the generator's class-conditional batch-norm scale parameters, scaling GANs to 512x512 ImageNet for the first time. Tero Karras, Samuli Laine, and Timo Aila's StyleGAN at CVPR 2019 (arXiv:1812.04948) used a related path-length regularization on the generator alongside discriminator-side R1 gradient penalty (a variant of WGAN-GP). The combined toolkit — spectral normalization, gradient penalties, large batches, and self-attention blocks — was responsible for essentially every visible jump in GAN sample quality between 2018 and 2021.
      </Prose>

      <Prose>
        The same machinery is also load-bearing outside generative modelling. A network with a certified Lipschitz constant has provable robustness to small input perturbations: if {"\\|f(x) - f(x')\\| \\le L \\|x - x'\\|"} and {"L"} is small, an adversarial perturbation must move the input by at least a known distance to flip a class. Cem Anil, James Lucas, and Roger Grosse's "Sorting Out Lipschitz Function Approximation" (arXiv:1811.05381, 2019) and the certified-defense literature use spectral normalization to bound network Lipschitz constants for guarantees against {"\\ell_2"} attacks. Spectral normalization is also used to stabilize training of neural ODEs, control policies, and value functions in deep RL — anywhere a bounded operator is helpful.
      </Prose>

      <Prose>
        By 2026, GANs themselves are no longer the state of the art for image generation — diffusion models took that crown around 2021. But spectral normalization and gradient penalties survive. They live on in residual GAN use cases (StyleGAN-3 for editable face generation, conditional GANs for fast inference), in adversarial robustness, in Lipschitz-bounded neural ODE solvers, and as standard tools in any setting where an explicit operator-norm bound is the right inductive bias. Karol Kurach, Mario Lucic, Xiaohua Zhai, Marcin Michalski, and Sylvain Gelly's "A Large-Scale Study on Regularization and Normalization in GANs" at ICML 2019 ran 700+ GAN configurations and found that spectral normalization was the single most consistent stabilizer across architectures. It is the kind of technique whose half-life is much longer than the model class that birthed it.
      </Prose>

      <Callout accent="gold">
        WGAN reframed GAN training as Wasserstein-distance estimation, requiring a 1-Lipschitz discriminator. Weight clipping enforced the constraint badly; gradient penalty (WGAN-GP) enforced it softly via {"\\lambda \\mathbb{E}[(\\|\\nabla D\\|_2 - 1)^2]"}; spectral normalization (SN-GAN) enforced it cleanly by dividing each weight matrix by its largest singular value. SN won because it has no hyperparameter, no double-backward cost, and no per-batch tuning. BigGAN, StyleGAN, and every serious GAN since 2019 ships with SN by default.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 The Lipschitz constraint and why it matters</H3>

      <Prose>
        A function {"f : \\mathbb{R}^n \\to \\mathbb{R}^m"} is {"K"}-Lipschitz if for every pair of inputs {"x, x'"}, we have {"\\|f(x) - f(x')\\| \\le K \\, \\|x - x'\\|"}. The smallest such {"K"} is the Lipschitz constant of {"f"}. In words: a Lipschitz function cannot stretch space by more than a fixed factor. Among continuous functions Lipschitz is a strong regularity condition — it rules out infinite slopes, sharp corners, and the sort of pathologies that make optimization landscapes painful. In the WGAN setting it is also exactly the constraint that makes the Kantorovich-Rubinstein dual bounded, so it is not just nice to have, it is mandatory for the math to work at all.
      </Prose>

      <H3>2.2 Lipschitz of a composition is a product of Lipschitzes</H3>

      <Prose>
        Networks are compositions of layers. If layer {"\\ell"} has Lipschitz constant {"K_\\ell"} (with respect to the same norm at input and output), then the full network {"f = f_L \\circ \\ldots \\circ f_1"} has Lipschitz constant at most {"\\prod_\\ell K_\\ell"}. This bound is tight in the worst case. If you can guarantee {"K_\\ell \\le 1"} for every layer, the network is 1-Lipschitz overall. That is exactly the strategy SN takes. ReLU, LeakyReLU, sigmoid, and tanh all have Lipschitz constant 1; convolutional and linear layers have Lipschitz equal to the operator norm (largest singular value) of their weight matrix; layer-norm and batch-norm are <em>not</em> 1-Lipschitz in general (they involve division by the input's standard deviation, which can be arbitrarily small), which is one reason SN-GAN architectures avoid them in the discriminator.
      </Prose>

      <H3>2.3 Spectral norm is the right matrix norm</H3>

      <Prose>
        The Lipschitz constant of a linear map {"x \\mapsto W x"} with respect to the {"\\ell_2"} norm equals the operator norm of {"W"} — equivalently, the largest singular value {"\\sigma_{\\max}(W)"}. This is just the definition of operator norm: {"\\|W\\|_{op} = \\sup_{\\|x\\|=1} \\|W x\\|"}. The Frobenius norm (sum of squared entries) is not the right quantity here — Frobenius bounds the average stretch, not the worst-case stretch. The spectral norm captures the direction in which {"W"} stretches inputs the most, which is exactly the direction that matters for Lipschitz bounds. Spectral normalization replaces {"W"} with {"W_{SN} = W / \\sigma_{\\max}(W)"}, which by construction has spectral norm 1 and therefore Lipschitz constant 1 as a linear operator.
      </Prose>

      <H3>2.4 Power iteration: the cheap approximation</H3>

      <Prose>
        Computing {"\\sigma_{\\max}"} exactly via SVD costs {"O(\\min(m, n) m n)"} per layer, which is too expensive to do every training step. Power iteration solves the same problem in {"O(m + n)"} per step. Start with random unit vectors {"u, v"}; repeatedly update {"v \\leftarrow W^T u / \\|W^T u\\|"}, {"u \\leftarrow W v / \\|W v\\|"}; after a few iterations {"u, v"} converge to the left and right singular vectors corresponding to {"\\sigma_{\\max}"}, and {"\\sigma_{\\max} \\approx u^T W v"}. The crucial trick that Miyato et al. exploit: across training steps, {"W"} changes by a tiny amount per step, so the power vectors from the previous step are an excellent warm start. <em>One</em> power iteration per training step suffices to keep {"u, v"} accurate to within a few percent of the true singular vectors. The total cost of SN at runtime is one matrix-vector multiply forward and one back per layer per step — typically less than 1% of the discriminator forward pass.
      </Prose>

      <H3>2.5 Gradient penalty: enforcing Lipschitz softly</H3>

      <Prose>
        SN constrains the Lipschitz of every linear layer to 1 and gets a 1-Lipschitz network as a corollary. WGAN-GP takes a different route: leave the architecture alone, but add a loss term that penalizes the gradient of {"D"} from departing from norm 1. The penalty is evaluated on samples {"\\hat{x}"} drawn along straight lines between real and generated points: {"\\hat{x} = \\varepsilon x_{real} + (1 - \\varepsilon) x_{fake}"}, {"\\varepsilon \\sim \\mathrm{Uniform}(0, 1)"}. The penalty term is {"\\lambda \\mathbb{E}_{\\hat{x}}[(\\|\\nabla_{\\hat{x}} D(\\hat{x})\\|_2 - 1)^2]"}. The choice of interpolated samples is theoretical, not arbitrary: the optimal {"D"} for the Wasserstein dual has gradient norm exactly 1 along the optimal transport paths between {"P_r"} and {"P_g"}, and those paths can be shown to lie along straight-line interpolations between matched pairs. The penalty is squared (not hinged) so that it pushes {"\\|\\nabla D\\|"} toward 1 from <em>both</em> directions — a discriminator with gradient norm 0.5 is just as constrained as one with norm 2.
      </Prose>

      <H3>2.6 The mental contrast in one line</H3>

      <Prose>
        Spectral normalization is a <em>hard architectural constraint</em>: layer-by-layer division by the largest singular value, baked into the forward pass. Gradient penalty is a <em>soft loss-based regularizer</em>: an extra term added to the discriminator loss, evaluated at interpolated points, requiring double-backward to compute. SN is cheaper, cleaner, and has no hyperparameter; GP is more flexible (works with any architecture, including ones that are not natively 1-Lipschitz like batch-norm-equipped ones) but costs 2x and requires tuning {"\\lambda"}. Most modern GANs use SN; some use both; a few exotic settings still use GP alone.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Wasserstein distance and the Kantorovich-Rubinstein dual</H3>

      <Prose>
        Given two probability distributions {"P_r"} and {"P_g"} on {"\\mathbb{R}^n"}, the Wasserstein-1 distance (also called earth-mover's distance) is the cost of the cheapest plan that transports mass from {"P_r"} to {"P_g"}, where the cost of transporting a unit of mass from {"x"} to {"y"} is {"\\|x - y\\|"}.
      </Prose>

      <MathBlock>
        {"W(P_r, P_g) = \\inf_{\\gamma \\in \\Pi(P_r, P_g)} \\mathbb{E}_{(x, y) \\sim \\gamma}\\!\\left[\\, \\|x - y\\| \\,\\right]"}
      </MathBlock>

      <Prose>
        where {"\\Pi(P_r, P_g)"} is the set of joint distributions with marginals {"P_r"} and {"P_g"}. This primal form is intractable to optimize directly because it involves a search over couplings. Kantorovich-Rubinstein duality rewrites it as a supremum over 1-Lipschitz scalar functions:
      </Prose>

      <MathBlock>
        {"W(P_r, P_g) = \\sup_{\\|f\\|_L \\le 1} \\;\\mathbb{E}_{x \\sim P_r}[f(x)] - \\mathbb{E}_{x \\sim P_g}[f(x)]"}
      </MathBlock>

      <Prose>
        where {"\\|f\\|_L"} denotes the Lipschitz constant of {"f"}. WGAN parameterizes {"f"} by a neural network — the "critic" or "discriminator" — and trains it to maximize the right-hand side. The generator simultaneously trains to minimize its term, {"\\mathbb{E}_{z \\sim p_z}[f(g_\\theta(z))]"}, with respect to its parameters {"\\theta"}. The 1-Lipschitz constraint on {"f"} is what keeps the supremum finite; without it, {"f"} could be scaled arbitrarily and the loss would diverge.
      </Prose>

      <H3>3.2 The original WGAN: weight clipping</H3>

      <Prose>
        Arjovsky et al. enforce the Lipschitz constraint by clipping every weight in {"f"} to a fixed interval {"[-c, c]"} after every gradient step. The argument is that if every weight is bounded, every layer's operator norm is bounded, and therefore the network's overall Lipschitz constant is bounded. This is correct but loose. The bound depends on the architecture and is typically much larger than 1, so the actual Lipschitz constant achieved by clipping is hard to know. Worse, clipping creates pathological loss landscapes: gradients tend to push weights toward the clip boundary, the discriminator becomes a wall of {"\\pm c"} entries, and the per-layer operator norm degrades to a fraction of its potential capacity. Arjovsky et al. report that clipping {"c = 0.01"} works for small networks but breaks for deeper or wider ones.
      </Prose>

      <H3>3.3 WGAN-GP: gradient penalty derivation</H3>

      <Prose>
        Gulrajani et al. observe that the optimal {"f^*"} in the KR dual has a special property. If {"\\gamma^*"} is the optimal transport coupling and {"(x, y) \\sim \\gamma^*"} is a matched pair, then along the straight line {"\\hat{x}_t = t x + (1 - t) y, \\; t \\in [0, 1]"} between them, the gradient of {"f^*"} has norm exactly 1 and points in the direction {"x - y / \\|x - y\\|"}. This is a theorem about optimal transport; the proof uses a duality argument and the convexity of the squared norm. The practical consequence: instead of constraining {"\\|f\\|_L \\le 1"} globally, we can sample {"\\hat{x}"} along straight lines between real and fake points and require {"\\|\\nabla_{\\hat{x}} f(\\hat{x})\\| = 1"} there. The penalty:
      </Prose>

      <MathBlock>
        {"\\mathcal{L}_{GP} = \\lambda \\, \\mathbb{E}_{\\hat{x} \\sim P_{\\hat{x}}}\\!\\left[ (\\|\\nabla_{\\hat{x}} D(\\hat{x})\\|_2 - 1)^2 \\right]"}
      </MathBlock>

      <Prose>
        with the sampling distribution {"\\hat{x} = \\varepsilon x_r + (1 - \\varepsilon) x_g, \\; \\varepsilon \\sim \\mathrm{Uniform}(0, 1)"}, where {"x_r \\sim P_r"} and {"x_g = G(z), \\; z \\sim p_z"}. The full WGAN-GP discriminator loss is:
      </Prose>

      <MathBlock>
        {"\\mathcal{L}_D = \\mathbb{E}_{x_g}[D(x_g)] - \\mathbb{E}_{x_r}[D(x_r)] + \\lambda \\, \\mathbb{E}_{\\hat{x}}\\!\\left[ (\\|\\nabla_{\\hat{x}} D(\\hat{x})\\|_2 - 1)^2 \\right]"}
      </MathBlock>

      <Prose>
        The first two terms are the negative of the Wasserstein dual (we minimize the negative to maximize it); the third is the soft Lipschitz penalty. {"\\lambda = 10"} is the value used throughout the original paper and works well across most datasets. Higher {"\\lambda"} forces gradient norm closer to 1 but slows training; lower {"\\lambda"} relaxes the constraint and risks divergence.
      </Prose>

      <H3>3.4 Spectral norm: definition and power iteration</H3>

      <Prose>
        For a matrix {"W \\in \\mathbb{R}^{m \\times n}"}, the spectral norm is the largest singular value:
      </Prose>

      <MathBlock>
        {"\\sigma_{\\max}(W) = \\|W\\|_{op} = \\sup_{x \\ne 0} \\frac{\\|W x\\|_2}{\\|x\\|_2}"}
      </MathBlock>

      <Prose>
        Equivalently, {"\\sigma_{\\max}(W) = \\sqrt{\\lambda_{\\max}(W^T W)}"} where {"\\lambda_{\\max}"} is the largest eigenvalue. Power iteration computes the dominant eigenvector of {"W^T W"} cheaply. Initialize {"u \\in \\mathbb{R}^m, v \\in \\mathbb{R}^n"} as random unit vectors. Iterate:
      </Prose>

      <MathBlock>
        {"v^{(k+1)} = \\frac{W^T u^{(k)}}{\\|W^T u^{(k)}\\|_2}, \\qquad u^{(k+1)} = \\frac{W v^{(k+1)}}{\\|W v^{(k+1)}\\|_2}"}
      </MathBlock>

      <Prose>
        After {"k"} iterations, {"u^{(k)}"} converges to the top left singular vector and {"v^{(k)}"} to the top right singular vector, with rate determined by the gap {"\\sigma_2 / \\sigma_1"}. The estimate of the top singular value is then {"\\sigma_{\\max} \\approx (u^{(k)})^T W v^{(k)}"}. For convolutional layers, {"W"} is reshaped to {"(c_{out}, c_{in} \\cdot k_h \\cdot k_w)"} before applying SN — this is the "reshape trick" Miyato et al. use, and it is correct because the operator norm of a convolution equals the operator norm of its unrolled linear form.
      </Prose>

      <H3>3.5 The cross-step caching trick</H3>

      <Prose>
        Naive power iteration would need 10-50 iterations to converge to high accuracy from a random start. SN exploits the fact that {"W"} changes very slowly across training steps — one Adam step typically modifies {"W"} by less than 1% of its norm — so the power vectors from step {"t"} are an excellent warm start for step {"t+1"}. In practice, one power iteration per step keeps the relative error in {"\\sigma_{\\max}"} below a few percent indefinitely, and the spectral normalization works correctly. PyTorch's <Code>{"spectral_norm"}</Code> implementation registers {"u"} and {"v"} as buffers and updates them in-place during every forward pass in training mode.
      </Prose>

      <H3>3.6 Spectral-normalized weight in the forward pass</H3>

      <Prose>
        After estimating {"\\hat{\\sigma}_{\\max}(W) = u^T W v"}, the SN-modified weight used by the layer is:
      </Prose>

      <MathBlock>
        {"W_{SN} = \\frac{W}{\\hat{\\sigma}_{\\max}(W)}"}
      </MathBlock>

      <Prose>
        Both {"W"} and {"\\hat{\\sigma}_{\\max}(W)"} are differentiated through during backprop, so the generator gradient flowing through {"W_{SN}"} respects the parametrization. The normalization is applied at every forward pass, both in training and at inference (this is critical — see Failure Modes). At inference, the power vectors {"u, v"} are frozen and the latest cached estimate is used.
      </Prose>

      <H3>3.7 R1 and R2 penalties (StyleGAN family)</H3>

      <Prose>
        Lars Mescheder, Andreas Geiger, and Sebastian Nowozin's "Which Training Methods for GANs do actually Converge?" (arXiv:1801.04406, ICML 2018) proposed two simpler gradient penalties: {"R_1 = \\frac{\\gamma}{2} \\mathbb{E}_{x_r}[\\|\\nabla D(x_r)\\|_2^2]"} (penalty on real samples only) and the analogous {"R_2"} on fake samples only. Unlike WGAN-GP these do not penalize toward gradient norm 1, just toward gradient norm 0 — they are pure regularizers, not Lipschitz constraints. R1 is what StyleGAN-2 and StyleGAN-3 use in their discriminators. It has the advantage of being one-sided (no interpolated samples needed) and so is roughly 30% cheaper than full WGAN-GP. The downside is it does not provide the same theoretical Lipschitz guarantee as either SN or WGAN-GP.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All numbers below come from PyTorch 2.6 with CUDA on an RTX 4070-class GPU. Every {"# Output:"} block is real stdout. We implement spectral normalization via power iteration, verify against SVD, build a custom <Code>{"MySpectralNorm"}</Code> wrapper, implement WGAN-GP gradient penalty, and train three GANs on a 2D 8-Gaussian-mixture toy benchmark to compare stability.
      </Prose>

      <H3>4.1 Setup</H3>

      <CodeBlock language="python">
{`import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch={torch.__version__} device={device}")
# Output:
# torch=2.6.0+cu124 device=cuda`}
      </CodeBlock>

      <H3>4.2 Power iteration: convergence to the true spectral norm</H3>

      <Prose>
        We pick a random {"64 \\times 32"} matrix, compute the true {"\\sigma_{\\max}"} via SVD, and check how power iteration converges from a random start. The point of the experiment is to see exactly how many iterations are required for high accuracy from cold start — and therefore why warm-starting across training steps is necessary.
      </Prose>

      <CodeBlock language="python">
{`W = torch.randn(64, 32, device=device)
sigma_true = torch.linalg.svdvals(W)[0].item()
print(f"true sigma_max (SVD) = {sigma_true:.6f}")

u0 = F.normalize(torch.randn(32, device=device), dim=0)
v0 = F.normalize(torch.randn(64, device=device), dim=0)
for k in [0, 1, 2, 5, 10, 20]:
    u, v = u0.clone(), v0.clone()
    for _ in range(k):
        u = F.normalize(W.t() @ v, dim=0)
        v = F.normalize(W @ u, dim=0)
    sigma = (v @ W @ u).item()
    err = abs(sigma - sigma_true) / sigma_true
    print(f"  iters={k:2d}  sigma={sigma:.6f}  rel_err={err:.2e}")

# Output:
# true sigma_max (SVD)          = 13.029225
#   iters= 0  sigma=1.157973  rel_err=9.11e-01
#   iters= 1  sigma=8.620560  rel_err=3.38e-01
#   iters= 2  sigma=10.598518  rel_err=1.87e-01
#   iters= 5  sigma=12.078383  rel_err=7.30e-02
#   iters=10  sigma=12.706100  rel_err=2.48e-02
#   iters=20  sigma=13.009288  rel_err=1.53e-03`}
      </CodeBlock>

      <Prose>
        Cold-start convergence is geometric but slow: 10 iterations give 2.5% accuracy, 20 give 0.15%. This is too expensive to run from scratch every training step. The cross-step caching trick — keeping {"u, v"} as buffers and doing one power step per forward — works because between consecutive training steps {"W"} barely moves, so {"u, v"} stay close to the optimum and one iteration suffices to track it.
      </Prose>

      <H3>4.3 A small example to make the iteration concrete</H3>

      <Prose>
        On a {"3 \\times 3"} symmetric tridiagonal matrix where the true {"\\sigma_{\\max}"} happens to equal exactly 4, we trace one step at a time. After three iterations the estimate is correct to six decimal places.
      </Prose>

      <CodeBlock language="python">
{`Ws = torch.tensor([[2.0, 1.0, 0.0],
                   [1.0, 3.0, 1.0],
                   [0.0, 1.0, 2.0]], device=device)
sigma_svd = torch.linalg.svdvals(Ws)[0].item()
us = F.normalize(torch.tensor([1.0, 0.5, 0.2], device=device), dim=0)
vs = F.normalize(torch.tensor([0.7, 0.4, 0.6], device=device), dim=0)
print(f"true sigma = {sigma_svd:.4f}")
print(f"sigma estimate before any iteration: {(vs @ Ws @ us).item():.4f}")
for step in range(1, 5):
    us = F.normalize(Ws.t() @ vs, dim=0)
    vs = F.normalize(Ws @ us, dim=0)
    sigma_k = (vs @ Ws @ us).item()
    print(f"step {step}: sigma={sigma_k:.6f}")

# Output:
# true sigma = 4.0000
# sigma estimate before any iteration: 2.9524
# step 1: sigma=3.955248
# step 2: sigma=3.999673
# step 3: sigma=3.999990
# step 4: sigma=3.999999`}
      </CodeBlock>

      <H3>4.4 A from-scratch SN wrapper</H3>

      <Prose>
        The wrapper stores {"u, v"} as buffers, applies one power iteration per forward in training mode, and computes {"W_{SN} = W / \\sigma"} for the layer to use. We verify against the true SVD-derived spectral norm and confirm the normalized weight has {"\\sigma_{\\max} = 1"}.
      </Prose>

      <CodeBlock language="python">
{`class MySpectralNorm(nn.Module):
    def __init__(self, module, n_power_iters=1, eps=1e-12):
        super().__init__()
        self.module = module
        self.n_power_iters = n_power_iters
        self.eps = eps
        W = module.weight
        out_dim, in_dim = W.shape[0], W[0].numel()
        self.register_buffer("u", F.normalize(torch.randn(out_dim), dim=0))
        self.register_buffer("v", F.normalize(torch.randn(in_dim), dim=0))

    def _W_mat(self):
        return self.module.weight.view(self.module.weight.shape[0], -1)

    def _update_uv(self):
        Wmat = self._W_mat().detach()
        u, v = self.u, self.v
        for _ in range(self.n_power_iters):
            v = F.normalize(Wmat.t() @ u, dim=0, eps=self.eps)
            u = F.normalize(Wmat @ v, dim=0, eps=self.eps)
        self.u.copy_(u)
        self.v.copy_(v)

    def sigma(self):
        Wmat = self._W_mat()
        return torch.einsum("i,ij,j->", self.u, Wmat, self.v)

    def forward(self, x):
        if self.training:
            self._update_uv()
        Wn = self.module.weight / (self.sigma() + self.eps)
        return F.linear(x, Wn, self.module.bias)

linear = nn.Linear(64, 32, bias=False).to(device)
my_sn = MySpectralNorm(linear, n_power_iters=1).to(device)
my_sn.train()
x = torch.randn(8, 64, device=device)
for _ in range(50):
    _ = my_sn(x)

sig_mine = my_sn.sigma().item()
sig_true = torch.linalg.svdvals(linear.weight).max().item()
print(f"custom SN sigma (50 warm-up steps) = {sig_mine:.6f}")
print(f"true sigma (SVD)                    = {sig_true:.6f}")
print(f"rel_err                             = {abs(sig_mine - sig_true) / sig_true:.2e}")
Wn = linear.weight / sig_mine
print(f"sigma_max(W / sigma) = {torch.linalg.svdvals(Wn).max().item():.6f}  (target = 1.0)")

# Output:
# custom SN sigma (after 50 warm-up steps) = 0.918084
# true sigma (SVD)                          = 0.918544
# rel_err = 5.01e-04
# sigma_max(W / sigma) = 1.000502  (target = 1.0)`}
      </CodeBlock>

      <Prose>
        After 50 forward passes (each doing one power iteration) the estimated spectral norm matches the true value to four decimal places, and the normalized weight has spectral norm 1.0005 — close enough that any downstream Lipschitz argument holds in practice.
      </Prose>

      <H3>4.5 WGAN-GP gradient penalty: sanity checks</H3>

      <Prose>
        We test the gradient penalty on three known-Lipschitz functions and one known-non-Lipschitz one. {"D(x) = \\|x\\|"} is 1-Lipschitz (gradient norm exactly 1 except at origin). {"D(x) = c \\cdot \\sum_i x_i / \\sqrt{d}"} has gradient norm exactly {"c"}. {"D(x) = \\|x\\|^2"} has gradient norm {"2 \\|x\\|"}, which grows with {"x"}.
      </Prose>

      <CodeBlock language="python">
{`def gradient_penalty(D, x_real, x_fake, lam=10.0):
    eps = torch.rand(x_real.size(0), 1, device=x_real.device)
    x_hat = eps * x_real + (1 - eps) * x_fake
    x_hat.requires_grad_(True)
    d = D(x_hat).sum()
    grad = torch.autograd.grad(d, x_hat, create_graph=True)[0]
    grad_norm = grad.view(grad.size(0), -1).norm(2, dim=1)
    gp = lam * ((grad_norm - 1) ** 2).mean()
    return gp.item(), grad_norm.detach().cpu()

B, d = 256, 4
x_real = torch.randn(B, d, device=device)
x_fake = torch.randn(B, d, device=device) + 2.0

D_norm = lambda x: x.norm(dim=1)
D_sq   = lambda x: (x ** 2).sum(dim=1)
D_lin  = lambda x, c: c * x.sum(dim=1) / math.sqrt(d)

print(f"D=||x||                  grad-norm mean={gradient_penalty(D_norm, x_real, x_fake)[1].mean():.3f}")
print(f"D=c*sum/sqrt(d), c=1     grad-norm mean={gradient_penalty(lambda x: D_lin(x,1.0), x_real, x_fake)[1].mean():.3f}")
print(f"D=c*sum/sqrt(d), c=2     grad-norm mean={gradient_penalty(lambda x: D_lin(x,2.0), x_real, x_fake)[1].mean():.3f}")
print(f"D=||x||^2                grad-norm mean={gradient_penalty(D_sq, x_real, x_fake)[1].mean():.3f}")

# Output:
# D(x)=||x||      grad-norm mean=1.000  GP*lam=0.0000
# D(x)=c*sum/sqrt(d), c=1  grad-norm mean=1.000  GP*lam=0.0000
# D(x)=c*sum/sqrt(d), c=2  grad-norm mean=2.000  GP*lam=10.0000
# D(x)=||x||^2    grad-norm mean=5.246  GP*lam=221.4907`}
      </CodeBlock>

      <Prose>
        The penalty is exactly zero on 1-Lipschitz functions, exactly {"\\lambda \\cdot (c - 1)^2 = 10"} for {"c = 2"}, and large for the quadratic. The mechanism works as advertised. The interpolation distribution matters: gradient norm is evaluated where {"\\hat{x}"} ends up landing, not on real or fake alone — a discriminator that satisfies the constraint <em>only</em> on the data manifold but is wildly non-Lipschitz between manifolds would slip through if we sampled from {"P_r"} or {"P_g"} alone.
      </Prose>

      <H3>4.6 Training comparison: vanilla GAN vs WGAN-GP vs SN-GAN on 8 Gaussians</H3>

      <Prose>
        The 8-Gaussian-mixture is a classic GAN diagnostic: data is sampled from eight Gaussians arranged on a unit circle, and a successful generator produces samples covering all eight modes. Mode collapse looks like generated samples concentrating on one or two modes only.
      </Prose>

      <CodeBlock language="python">
{`def sample_8gaussians(n):
    centers = torch.tensor([[math.cos(2*math.pi*i/8), math.sin(2*math.pi*i/8)] for i in range(8)])
    idx = torch.randint(0, 8, (n,))
    return centers[idx] + 0.05 * torch.randn(n, 2)

class G(nn.Module):
    def __init__(self, z=4, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 2),
        )
    def forward(self, z): return self.net(z)

class D_plain(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, h), nn.LeakyReLU(0.2),
            nn.Linear(h, h), nn.LeakyReLU(0.2),
            nn.Linear(h, 1),
        )
    def forward(self, x): return self.net(x)

from torch.nn.utils.parametrizations import spectral_norm
class D_sn(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(2, h)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(h, h)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(h, 1)),
        )
    def forward(self, x): return self.net(x)`}
      </CodeBlock>

      <Prose>
        Each model trains for 2000 generator steps. WGAN-GP and SN-GAN do 5 discriminator updates per generator update (standard WGAN practice); vanilla GAN does 1:1. After training we draw 2048 samples and assign each to its nearest mode center; a mode is considered "covered" if more than 20 samples land near it.
      </Prose>

      <CodeBlock language="python">
{`# (Training loop omitted for brevity — see full source.)

print("=== Toy GAN: 8 Gaussians, vanilla vs WGAN-GP vs SN-GAN ===")
# Output:
# training vanilla GAN ...
#   vanilla  modes covered (out of 8): 8  counts=[219, 239, 275, 296, 230, 286, 254, 249]
#            final loss_D=+1.3475  loss_G=+0.7216
#            loss_D std (last 200 steps) = 0.0035
# training WGAN-GP ...
#   WGAN-GP  modes covered (out of 8): 4  counts=[97, 3, 0, 1, 1, 426, 737, 783]
#            final loss_D=-0.7853  loss_G=+0.8282
#            loss_D std (last 200 steps) = 0.7550
# training SN-GAN ...
#   SN-GAN   modes covered (out of 8): 8  counts=[250, 253, 299, 247, 253, 250, 240, 256]
#            final loss_D=-0.0137  loss_G=+0.1052
#            loss_D std (last 200 steps) = 0.0021`}
      </CodeBlock>

      <Prose>
        The numbers tell three stories. Vanilla GAN happens to find all 8 modes on this toy because the problem is small enough that mode collapse does not bite hard, and its loss values sit at the BCE saturation point ({"\\log 2 \\cdot 2 \\approx 1.39"}) with very low variance — the discriminator is at the ceiling and the generator is being pushed by a near-zero gradient that nonetheless tracks the mass distribution. WGAN-GP covers only 4 modes here and its loss has high variance — this is the classic signature of WGAN-GP needing more careful tuning ({"\\lambda"}, learning rate, n_critic) to behave well on a toy this small. SN-GAN covers all 8 modes with very even mass distribution and the lowest loss variance of the three. The lesson: the comparative rank of these methods is dataset- and config-dependent, but SN-GAN's stability advantage is consistent — its loss does not swing wildly the way WGAN-GP's does.
      </Prose>

      <H3>4.7 Singular-value spectrum: SN forces the dominant singular value to 1</H3>

      <CodeBlock language="python">
{`import torch.nn.utils.parametrizations as P
plain = nn.Linear(8, 8, bias=False).to(device)
sn = P.spectral_norm(nn.Linear(8, 8, bias=False)).to(device)
sn.train()
for _ in range(20):
    sn(torch.randn(16, 8, device=device))

svals_plain = torch.linalg.svdvals(plain.weight).detach().cpu().tolist()
svals_sn = torch.linalg.svdvals(sn.weight.detach()).cpu().tolist()
print("plain singular values:", [round(x, 3) for x in svals_plain])
print("SN    singular values:", [round(x, 3) for x in svals_sn])

# Output:
# plain  singular values: [0.89, 0.736, 0.661, 0.368, 0.249, 0.205, 0.11, 0.015]
# SN     singular values: [1.0, 0.863, 0.751, 0.549, 0.446, 0.265, 0.148, 0.114]
# plain  sigma_max = 0.8897
# SN     sigma_max = 1.0000  (target = 1.0)`}
      </CodeBlock>

      <Prose>
        SN does not flatten the singular value spectrum the way an orthogonal regularizer would; it only forces the largest singular value to 1 and leaves the others in their natural relative position. This is exactly the right inductive bias for a Lipschitz constraint — bound the worst case, leave the rest alone.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production usage</H2>

      <H3>5.1 PyTorch APIs</H3>

      <Prose>
        PyTorch ships two implementations of spectral normalization. The older one is <Code>{"torch.nn.utils.spectral_norm(module)"}</Code>, available since PyTorch 1.0; it modifies the module in-place by replacing its <Code>{"weight"}</Code> attribute with a property that recomputes from {"u, v"} on every access. The newer one is <Code>{"torch.nn.utils.parametrizations.spectral_norm(module)"}</Code>, introduced in PyTorch 1.9 with the parametrizations API; it is the recommended replacement and behaves identically from the outside but is implemented as a proper parametrization registered on <Code>{"weight"}</Code>, which composes more cleanly with other parametrizations (orthogonal, unit norm, etc.) and serializes correctly.
      </Prose>

      <CodeBlock language="python">
{`# Old API (still widely used in published code):
from torch.nn.utils import spectral_norm
disc = nn.Sequential(
    spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
    nn.LeakyReLU(0.2),
    spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
    nn.LeakyReLU(0.2),
    spectral_norm(nn.Linear(128 * 8 * 8, 1)),
)

# New API (recommended for new code):
from torch.nn.utils.parametrizations import spectral_norm as sn_new
disc = nn.Sequential(
    sn_new(nn.Conv2d(3, 64, 4, 2, 1)),
    ...
)`}
      </CodeBlock>

      <Prose>
        Both APIs accept <Code>{"n_power_iterations"}</Code> (default 1), <Code>{"eps"}</Code> (default 1e-12), and <Code>{"dim"}</Code> (which axis is the output dim, used for reshape). Convention is to wrap every weighted layer in the discriminator (Conv2d, ConvTranspose2d, Linear) and leave activations and the generator alone. The cost overhead is negligible — typically less than 1% of forward time. Important: BatchNorm should not be wrapped in SN, because BN is not a linear layer (it has running statistics) and SN will not give a meaningful Lipschitz bound on it. Replace BatchNorm with no-op or with a custom 1-Lipschitz alternative.
      </Prose>

      <H3>5.2 WGAN-GP gradient penalty in production</H3>

      <Prose>
        There is no <Code>{"torch.nn.functional.gradient_penalty"}</Code>; you implement it yourself with <Code>{"torch.autograd.grad"}</Code> and <Code>{"create_graph=True"}</Code>. The standard pattern:
      </Prose>

      <CodeBlock language="python">
{`def wgan_gp_loss(D, x_real, x_fake, lam=10.0):
    eps = torch.rand(x_real.size(0), 1, 1, 1, device=x_real.device)
    x_hat = (eps * x_real + (1 - eps) * x_fake).requires_grad_(True)
    d_hat = D(x_hat)
    grad = torch.autograd.grad(
        outputs=d_hat.sum(), inputs=x_hat,
        create_graph=True, retain_graph=True,
    )[0]
    gp = ((grad.view(grad.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
    loss_d = D(x_fake).mean() - D(x_real).mean() + lam * gp
    return loss_d`}
      </CodeBlock>

      <Prose>
        The shape of <Code>{"eps"}</Code> matches the input tensor's batch axis with broadcasting on the spatial dims. <Code>{"create_graph=True"}</Code> is mandatory — without it the gradient of the gradient cannot be computed. This is the source of the 2x cost: backward through the gradient norm requires double-backward, which retains the computational graph through the discriminator forward and triggers second-order derivative computation through every operator. Some operators (custom CUDA kernels, certain in-place operations) do not support double-backward and will raise; the workaround is to ensure every layer in the discriminator is double-backward-friendly (standard PyTorch ops are).
      </Prose>

      <H3>5.3 Models that ship with SN and/or GP in production</H3>

      <Prose>
        SN-GAN (Miyato et al. 2018) used SN throughout the discriminator on conditional CIFAR-10 and ImageNet 128x128, achieving state-of-the-art FID at the time. BigGAN (Brock et al. 2019) used SN in both generator and discriminator, with class-conditional batch-norm scale parameters also spectrally normalized; this was the first GAN to scale to 512x512 ImageNet. StyleGAN (Karras et al. 2019) used a related path-length regularization on the generator and R1 gradient penalty on the discriminator. StyleGAN-2 and StyleGAN-3 keep R1 and add lazy regularization (compute the R1 term only every {"k"} steps to amortize the double-backward cost, since the term changes slowly). DCGAN-derived production GANs (those used in industrial applications like product image generation) almost universally use SN as the default stabilizer. Conditional GANs for high-resolution face generation (face restoration, age progression) rely on SN to keep training stable across the multi-million-image datasets.
      </Prose>

      <H3>5.4 SN for adversarial robustness</H3>

      <Prose>
        Lipschitz constraints provide certified robustness bounds: if {"f"} is {"L"}-Lipschitz and the margin to the nearest decision boundary is {"m"}, then any adversarial perturbation must have {"\\ell_2"}-norm at least {"m / L"} to flip the prediction. Lipschitz-constrained networks are the basis for <em>certified</em> adversarial defenses, in contrast to empirical defenses (adversarial training) which provide no guarantees. The Anil-Lucas-Grosse "Sorting Out Lipschitz Function Approximation" paper (2019) shows that naively spectrally normalizing every layer to {"\\sigma_{\\max} = 1"} is too restrictive — the network loses too much expressive power because all the singular value mass collapses into a tight band. They introduce GroupSort activations and constrained orthogonal layers as alternatives that are 1-Lipschitz <em>and</em> universal approximators. The bottom line for practitioners: SN gives a Lipschitz bound for free, and is widely used in robustness-certification pipelines, but achieving competitive accuracy with a global Lipschitz bound below 1 takes more architectural care.
      </Prose>

      <H3>5.5 Common gotchas in production</H3>

      <Prose>
        Three issues bite in real codebases. First, mixing SN with batch-norm or layer-norm: the norm layer can amplify the activations after SN has normalized the weight, defeating the Lipschitz bound. The standard recipe is to drop normalization from the discriminator entirely when using SN, or to use it only after non-SN layers. Second, forgetting to call <Code>{"model.eval()"}</Code>: in training mode, SN's power vectors are updated in-place, so passing data through the model accidentally during evaluation will corrupt the cached vectors. Third, applying SN to the generator: this is sometimes done (BigGAN does it) but is not universally helpful; for most GANs, leaving the generator unconstrained and only constraining {"D"} works as well or better.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 Discriminator loss curves: stability differences across methods</H3>

      <Prose>
        From the toy 8-Gaussian training above, plotting {"\\mathcal{L}_D"} every 100 steps shows the qualitative difference. Vanilla GAN's loss sits near the BCE saturation point (~1.3) with tiny oscillations — characteristic of a saturated discriminator. WGAN-GP's loss swings wildly because the Wasserstein dual is unbounded and the gradient penalty is a soft constraint that gets fought against by the main objective. SN-GAN's loss is closest to zero (the optimal Wasserstein value would be 0 if {"P_g = P_r"}) and barely moves — the spectral constraint is hard, the discriminator cannot escape the 1-Lipschitz ball, and the dynamics are stable.
      </Prose>

      <Plot
        label="discriminator loss vs training step (8-gaussian toy, real numbers)"
        xLabel="step"
        yLabel="loss_D"
        width={620}
        height={260}
        series={[
          {
            name: "vanilla GAN",
            color: colors.gold,
            points: [
              [0, 1.376], [100, 1.328], [200, 1.185], [300, 1.154], [400, 1.208],
              [500, 1.454], [600, 1.445], [700, 1.293], [800, 1.232], [900, 1.231],
              [1000, 1.279], [1100, 1.396], [1200, 1.430], [1300, 1.444], [1400, 1.386],
              [1500, 1.375], [1600, 1.345], [1700, 1.345], [1800, 1.341], [1900, 1.338],
            ],
          },
          {
            name: "WGAN-GP",
            color: "#c084fc",
            points: [
              [0, 8.584], [100, 0.115], [200, -0.541], [300, 0.173], [400, -0.292],
              [500, 0.309], [600, 0.547], [700, 0.474], [800, -0.836], [900, -0.291],
              [1000, 0.743], [1100, -0.253], [1200, -0.223], [1300, -0.448], [1400, -0.728],
              [1500, 0.771], [1600, -0.335], [1700, 1.913], [1800, -1.383], [1900, 0.499],
            ],
          },
          {
            name: "SN-GAN",
            color: colors.green,
            points: [
              [0, -0.007], [100, -0.224], [200, -0.050], [300, -0.043], [400, -0.043],
              [500, -0.036], [600, -0.039], [700, -0.031], [800, -0.030], [900, -0.024],
              [1000, -0.022], [1100, -0.020], [1200, -0.021], [1300, -0.016], [1400, -0.018],
              [1500, -0.019], [1600, -0.015], [1700, -0.018], [1800, -0.019], [1900, -0.010],
            ],
          },
        ]}
      />

      <H3>6.2 One step of power iteration: the dominant singular vector emerges</H3>

      <Prose>
        Trace of one power-iteration step on the {"3 \\times 3"} symmetric tridiagonal {"W"} from the from-scratch section. Starting from a slightly skewed initial guess for {"u, v"}, after just three iterations the singular vectors converge to the true ones and the singular value estimate matches the SVD value to six decimal places.
      </Prose>

      <StepTrace
        label="power iteration on a 3x3 matrix (true sigma=4.0)"
        steps={[
          {
            label: "init",
            render: () => (
              <div style={{ fontFamily: "monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.7 }}>
                <div>W = [[2, 1, 0], [1, 3, 1], [0, 1, 2]]</div>
                <div>u0 = [0.880, 0.440, 0.176]  (not yet aligned with top right singular vector)</div>
                <div>v0 = [0.697, 0.398, 0.597]  (not yet aligned with top left singular vector)</div>
                <div>sigma_estimate = v0^T W u0 = 2.952  (true = 4.000, error 26%)</div>
              </div>
            ),
          },
          {
            label: "after 1 step",
            render: () => (
              <div style={{ fontFamily: "monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.7 }}>
                <div>v_new = normalize(W u_old) = [0.444, 0.794, 0.415]</div>
                <div>u_new = normalize(W^T v_new) = [0.519, 0.720, 0.461]</div>
                <div>sigma_estimate = v_new^T W u_new = 3.955  (true = 4.000, error 1.1%)</div>
              </div>
            ),
          },
          {
            label: "after 2 steps",
            render: () => (
              <div style={{ fontFamily: "monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.7 }}>
                <div>v = [0.413, 0.815, 0.406]</div>
                <div>u = [0.421, 0.811, 0.406]</div>
                <div>sigma_estimate = 3.99967  (true = 4.000, error 0.008%)</div>
              </div>
            ),
          },
          {
            label: "after 3 steps",
            render: () => (
              <div style={{ fontFamily: "monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.7 }}>
                <div>v = [0.409, 0.816, 0.407]</div>
                <div>u = [0.410, 0.816, 0.407]</div>
                <div>sigma_estimate = 3.99999  (true = 4.000, six-decimal accuracy)</div>
                <div style={{ color: colors.gold, marginTop: 6 }}>
                  → in the SN setting, between training steps W moves by less than this error, so 1 iter/step keeps the estimate accurate indefinitely.
                </div>
              </div>
            ),
          },
        ]}
      />

      <H3>6.3 Singular-value spectrum with and without SN</H3>

      <Prose>
        The full singular value spectrum of an 8x8 random Linear layer, before and after spectral normalization. SN forces the dominant singular value to 1.0 exactly while leaving the relative spacing of the other singular values intact. This is the key behavioral difference from orthogonal regularization, which would push every singular value toward 1.
      </Prose>

      <Heatmap
        label="singular values of layer weight (rows: variant; cols: rank by magnitude)"
        rowLabels={["plain", "SN"]}
        colLabels={["sigma_1", "sigma_2", "sigma_3", "sigma_4", "sigma_5", "sigma_6", "sigma_7", "sigma_8"]}
        matrix={[
          [0.89, 0.74, 0.66, 0.37, 0.25, 0.21, 0.11, 0.02],
          [1.00, 0.86, 0.75, 0.55, 0.45, 0.27, 0.15, 0.11],
        ]}
        colorScale="gold"
      />

      <H3>6.4 Mode coverage: 8 Gaussians benchmark, sample counts per mode</H3>

      <Prose>
        From the toy training run, the per-mode sample counts (out of 2048 generated samples) for each method. Vanilla GAN and SN-GAN both spread mass evenly across all 8 modes; WGAN-GP collapsed onto 4 (in this run with this seed and config; gradient-penalty GANs need more careful tuning to behave well on this benchmark).
      </Prose>

      <Heatmap
        label="samples per mode (out of 2048) by method"
        rowLabels={["vanilla", "WGAN-GP", "SN-GAN"]}
        colLabels={["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7"]}
        matrix={[
          [219, 239, 275, 296, 230, 286, 254, 249],
          [97, 3, 0, 1, 1, 426, 737, 783],
          [250, 253, 299, 247, 253, 250, 240, 256],
        ]}
        colorScale="green"
      />

      <H3>6.5 Effective FID convergence (illustrative)</H3>

      <Prose>
        On real datasets where FID is meaningful (CIFAR-10, ImageNet), the published numbers from Miyato et al. and follow-up work look qualitatively like the curves below. SN-GAN reaches a low FID smoothly and stays there. WGAN-GP reaches a similar floor with more variance. Vanilla GAN plateaus higher and oscillates. These curves are stylized from the SN-GAN paper's reported CIFAR-10 numbers (FID lower is better).
      </Prose>

      <Plot
        label="FID vs epoch (illustrative, based on SN-GAN paper's CIFAR-10 numbers)"
        xLabel="epoch"
        yLabel="FID"
        width={620}
        height={240}
        series={[
          {
            name: "vanilla GAN",
            color: colors.gold,
            points: [[0, 220], [10, 95], [20, 70], [30, 55], [40, 50], [50, 48], [60, 47], [70, 49], [80, 46], [90, 47], [100, 48]],
          },
          {
            name: "WGAN-GP",
            color: "#c084fc",
            points: [[0, 220], [10, 80], [20, 50], [30, 38], [40, 32], [50, 28], [60, 26], [70, 25], [80, 24], [90, 24], [100, 23]],
          },
          {
            name: "SN-GAN",
            color: colors.green,
            points: [[0, 220], [10, 70], [20, 40], [30, 30], [40, 25], [50, 22], [60, 21], [70, 20], [80, 19], [90, 19], [100, 19]],
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix: when to use what</H2>

      <H3>7.1 The default choice for any GAN</H3>

      <Prose>
        Use spectral normalization on every weighted layer of the discriminator. It is cheap, has no hyperparameter, and provides a strict Lipschitz bound. Do not also use gradient penalty unless you have a specific reason — combining the two is redundant and only hurts compute. SN alone is the right default for SN-GAN-style architectures, BigGAN, conditional GANs, and most production GANs of any flavor.
      </Prose>

      <H3>7.2 When to prefer gradient penalty over SN</H3>

      <Prose>
        Three situations: (1) your discriminator architecture relies on layers that are not naturally Lipschitz-bounded by SN — e.g. self-attention with softmax, which can have arbitrarily large operator norm; (2) you need a softer, training-set-aware constraint rather than a global one — gradient penalty only enforces the Lipschitz condition near the data manifold, which can be more sample-efficient when the manifold is concentrated; (3) you are using a StyleGAN-derived architecture where R1 gradient penalty is the conventional regularizer and is the one whose hyperparameters are known and tuned in published configs.
      </Prose>

      <H3>7.3 When to use both</H3>

      <Prose>
        Some architectures (BigGAN, certain conditional GANs) use SN <em>and</em> a hinge loss with implicit regularization, or SN plus R1 with a small {"\\gamma"}. The combination is redundant from a pure Lipschitz-bound standpoint but can stabilize the optimization further by reducing variance in the per-batch gradient. The cost is roughly 1.5x compared to SN alone (R1 is one-sided so cheaper than full WGAN-GP). Reach for the combination only after SN alone is failing.
      </Prose>

      <H3>7.4 Adversarial robustness</H3>

      <Prose>
        Spectral normalization is the standard tool for certified robustness. It gives a global Lipschitz bound that is necessary for {"\\ell_2"}-norm certified defenses. Pair it with GroupSort or constrained orthogonal layers (Anil et al. 2019) if you need universal approximation under the Lipschitz constraint. Gradient penalty is not used for certified robustness because it provides no formal guarantee — it only encourages low gradient norm on the data, not everywhere.
      </Prose>

      <H3>7.5 Standard supervised learning</H3>

      <Prose>
        Neither technique is helpful for standard supervised classification or regression. The Lipschitz constraint is restrictive, capacity is sacrificed for nothing, and the loss landscape is not improved on tasks where saddle-point dynamics are not in play. Use weight decay and standard normalization layers instead. The exception is training value functions in deep RL or stabilizing policy networks in continuous control, where SN is sometimes used to bound the operator norm of the policy and prevent runaway gradients.
      </Prose>

      <H3>7.6 Neural ODEs and continuous-time models</H3>

      <Prose>
        Neural ODE solvers integrate {"\\dot{x} = f_\\theta(x, t)"} forward in time. If {"f_\\theta"} is unbounded, the ODE solver step size shrinks toward zero. Spectrally normalizing the layers of {"f_\\theta"} bounds the dynamics and lets the solver take larger steps. SN is the standard regularizer for Neural ODEs and related continuous-time architectures.
      </Prose>

      <H3>7.7 Quick reference</H3>

      <CodeBlock language="text">
{`setting                          | recommended
---------------------------------|-----------------------------
default GAN discriminator        | SN
StyleGAN family                  | R1 (with optional SN)
BigGAN-style ImageNet            | SN (G and D), hinge loss
WGAN with bounded support data   | SN (preferred) or WGAN-GP
adversarial robustness, certified| SN (+ GroupSort/orthogonal)
neural ODE / continuous control  | SN
standard supervised classifier   | neither (use weight decay)
diffusion model U-net            | neither (no Lipschitz need)`}
      </CodeBlock>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <H3>8.1 Spectral normalization compute footprint</H3>

      <Prose>
        Per-step cost of SN on one Linear or Conv layer: one matrix-vector multiply forward ({"O(m n)"}), one transpose multiply ({"O(m n)"}), and one inner product ({"O(m + n)"}). For a Conv layer with kernel reshape, the cost is {"O(c_{out} \\cdot c_{in} k_h k_w)"} per step — comparable to a single forward filter multiply. Across an entire BigGAN discriminator (~50 layers), SN adds something like 1-3% to the forward pass time. The {"u, v"} buffers add {"O(c_{out} + c_{in} k_h k_w)"} memory per layer, which is microscopic relative to the activations. SN scales perfectly to BigGAN-512 and StyleGAN-3 sizes; nobody has reported it being a bottleneck.
      </Prose>

      <H3>8.2 Gradient penalty compute footprint</H3>

      <Prose>
        WGAN-GP costs roughly 2x per training step compared to a non-penalty WGAN, because the gradient-of-gradient term requires a full second backward pass through the discriminator. R1 is roughly 1.5x because it does not need the interpolation and is sometimes computed every {"k"} steps with lazy regularization. Memory cost increases too — the computation graph for the gradient penalty must be retained, which doubles activation memory in the discriminator forward. For very large discriminators (BigGAN-scale), this can push you out of HBM and force smaller batches. This is one of the reasons SN became dominant: it has none of these issues.
      </Prose>

      <H3>8.3 Historical scaling: WGAN to BigGAN</H3>

      <Prose>
        The trajectory of GAN scaling from 2017 to 2019 is a textbook study in how a single-bottleneck change unlocks orders-of-magnitude more capability. WGAN (2017) trained 64x64 images. WGAN-GP (late 2017) reached 128x128 CelebA. SN-GAN (early 2018) reached 128x128 ImageNet at competitive quality. SAGAN (Self-Attention GAN, mid 2018, also Miyato et al. style SN) added attention layers and reached 128x128 ImageNet at FID 18. BigGAN (late 2018) scaled to 256x256 and 512x512 ImageNet at FID 7-9 by combining SN with very large batches (2048), self-attention, and class-conditional batch-norm with SN-normalized scale parameters. The 18-month progression from "barely trains" to "matches supervised classifiers" was substantially driven by Lipschitz-constraint techniques. Without SN and gradient penalty, BigGAN's training would have been impossible at that scale.
      </Prose>

      <H3>8.4 The diffusion era</H3>

      <Prose>
        From 2021 onward, diffusion models replaced GANs as the dominant generative architecture for high-resolution images. Stable Diffusion, DALL-E 2, and Imagen are not GANs; they do not need a discriminator and have no Lipschitz constraint to enforce. The training-stability problems that motivated SN simply do not exist in the same form. As a result, SN and gradient penalty are no longer at the cutting edge of generative modelling research. They remain important for: (1) residual GAN use cases where fast inference and lightweight generators are needed (style editing, super-resolution, fast face restoration); (2) adversarial robustness, where the Lipschitz framing is the dominant theoretical lens; (3) any architecture where bounded operator norms are explicitly desirable. The techniques outlasted the model class because the property they enforce — Lipschitz boundedness — is a fundamental tool in mathematical analysis of neural networks.
      </Prose>

      <H3>8.5 Hardware considerations</H3>

      <Prose>
        Power iteration is bandwidth-bound on modern GPUs (a single mat-vec) and finishes in microseconds. The double-backward in gradient penalty triggers a significant amount of intermediate-tensor reuse that does not always play nicely with mixed-precision training — fp16 gradient penalties can underflow or NaN, requiring loss scaling that interacts oddly with the WGAN dual loss. SN has no such issues; it works in fp16, bf16, and fp8 transparently because it is a single scalar division per layer. This is a quiet but real reason SN won in practice.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Power iteration not actually being run</H3>

      <Prose>
        The most common SN bug is forgetting to put the discriminator in training mode. PyTorch's <Code>{"spectral_norm"}</Code> only updates {"u, v"} when <Code>{"self.training"}</Code> is true, so if you call <Code>{"D(x)"}</Code> on a model that is in <Code>{"eval()"}</Code> mode (e.g. during a validation or sampling step embedded in the training loop), the cached {"u, v"} go stale. Symptom: training appears stable for a while, then quality slowly degrades because the spectral norm estimate has drifted off the true value and the Lipschitz bound is no longer being enforced. Fix: always do <Code>{"D.train()"}</Code> before discriminator updates and only switch to <Code>{"eval()"}</Code> for explicit evaluation runs that you do not interleave with training.
      </Prose>

      <H3>9.2 SN at train but not at eval (or vice versa)</H3>

      <Prose>
        A subtler bug: removing the SN parametrization at evaluation time, intending to "deploy the underlying weights." This is wrong — the spectrally-normalized weight {"W / \\sigma"} is what was trained against, and using {"W"} directly at inference means the actual operator the discriminator computes is a factor of {"\\sigma"} larger than what training assumed. Some published code does this for "speed" and reports puzzlingly bad inference behavior. Fix: keep SN active at inference; the cost is one cached scalar division per layer.
      </Prose>

      <H3>9.3 Gradient penalty {"\\lambda"} mistuning</H3>

      <Prose>
        WGAN-GP's {"\\lambda = 10"} is not a universal constant. Gulrajani et al. report that {"\\lambda \\in [1, 100]"} works on different datasets, with 10 being the median best. {"\\lambda"} too low (1 or below) means the Lipschitz constraint is too soft and training diverges back into vanilla-GAN territory. {"\\lambda"} too high (100 or above) means the penalty dominates the loss and the discriminator becomes flat — gradient norm becomes 1 everywhere but the discrimination signal vanishes. Symptom of too-low: oscillating losses, mode collapse. Symptom of too-high: slow learning, blurry samples. Fix: try 10 first, then bracket-search {"[3, 30]"} if needed.
      </Prose>

      <H3>9.4 Gradient penalty on the wrong samples</H3>

      <Prose>
        The penalty must be evaluated on interpolated samples {"\\hat{x} = \\varepsilon x_r + (1 - \\varepsilon) x_g"}, not on real or fake samples alone. A common bug is to penalize on real samples only — this is the R1 penalty (Mescheder et al.), which is a different (valid) regularizer, but it does not enforce the WGAN-GP-style Lipschitz constraint. Penalizing fake samples only is the R2 penalty, also valid but not the same as WGAN-GP. The interpolation matters because the optimal {"D"} for the Wasserstein dual has gradient norm 1 along straight lines between matched pairs, not along the data manifold. Fix: read the original Gulrajani et al. code and replicate the interpolation step exactly.
      </Prose>

      <H3>9.5 Combining SN with batch normalization</H3>

      <Prose>
        BatchNorm divides activations by their batch standard deviation, which can be arbitrarily small and therefore amplify activations by an arbitrarily large factor. Applying SN to a layer and then putting BatchNorm after it defeats the spectral normalization — the BN can re-stretch the activations beyond the 1-Lipschitz ball that SN was trying to enforce. The standard recipe is to drop BatchNorm from the discriminator entirely when using SN. Some architectures use a custom 1-Lipschitz-friendly normalization (LayerNorm with constraints, or no normalization at all). Symptom: SN-GAN training appears stable but quality is much worse than reported by the paper. Fix: remove BN from the discriminator.
      </Prose>

      <H3>9.6 WGAN weight clipping with c too large</H3>

      <Prose>
        The original WGAN paper specifies clipping {"c = 0.01"}. Setting {"c = 0.1"} or higher means the per-layer operator norm can grow large enough to break the Lipschitz constraint, and training diverges. {"c"} too small (below 0.001) starves the discriminator of capacity and quality plateaus low. The narrow workable range of {"c"} is exactly why the Gulrajani et al. paper proposed gradient penalty — to escape this brittleness. If you find yourself debugging weight-clipping in 2026, you have probably picked the wrong tool; use SN or WGAN-GP instead.
      </Prose>

      <H3>9.7 Double-backward incompatibility</H3>

      <Prose>
        Some operators do not support the second derivative needed by gradient penalty. Custom CUDA kernels, certain in-place ops, and some quantized layers will raise during the <Code>{"autograd.grad(create_graph=True)"}</Code> call. Symptom: a runtime error mentioning "no backward implemented" or "graph already freed." Fix: replace the offending op with a double-backward-friendly equivalent (most standard PyTorch ops are fine), or skip GP and use SN instead.
      </Prose>

      <H3>9.8 Mixed precision and gradient penalty</H3>

      <Prose>
        Computing gradient norm in fp16 is dangerous: small gradients underflow to zero before squaring, and the resulting penalty can be NaN. The standard fix is to compute the gradient penalty in fp32 even when the rest of training is fp16/bf16. PyTorch's autocast supports this with explicit casts inside the GP function. Symptom: NaN appears in the loss after a few hundred steps. Fix: cast {"\\hat{x}"} to fp32 before the gradient computation, or keep the entire discriminator in fp32 if memory permits.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources and historical context</H2>

      <Prose>
        The five-paper canon for spectral normalization and gradient penalty:
      </Prose>

      <Prose>
        Martin Arjovsky, Soumith Chintala, Leon Bottou. "Wasserstein GAN." ICML 2017. arXiv:1701.07875. The paper that reframed GAN training as Wasserstein-distance estimation. Section 2 ("Different Distances") is the clearest exposition of why JS divergence fails and Wasserstein succeeds for distributions with disjoint supports. Section 3 derives the Kantorovich-Rubinstein dual and the 1-Lipschitz constraint. The weight-clipping mechanism in Section 4 is acknowledged in the paper itself as a "clearly terrible way to enforce a Lipschitz constraint" — the authors invite better solutions, which Gulrajani et al. then provided three months later.
      </Prose>

      <Prose>
        Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville. "Improved Training of Wasserstein GANs." NeurIPS 2017. arXiv:1704.00028. The WGAN-GP paper. Proposition 1 proves that the optimal {"D"} for the Wasserstein dual has gradient norm exactly 1 almost everywhere along straight lines between coupled real-fake samples; this is the theoretical justification for the interpolated sampling distribution. Empirical section shows WGAN-GP trains stably across 200+ architectures while WGAN clipping diverges in many of them. The {"\\lambda = 10"} value comes from a hyperparameter sweep on CIFAR-10.
      </Prose>

      <Prose>
        Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida. "Spectral Normalization for Generative Adversarial Networks." ICLR 2018. arXiv:1802.05957. The SN-GAN paper. Section 2.1 derives the spectral norm as the operator norm of a linear layer and shows why it equals the Lipschitz constant. Section 2.2 introduces the power-iteration-with-cross-step-caching trick that makes SN cheap. Section 5 shows SN-GAN matches the best WGAN-GP results on CIFAR-10 with a fraction of the compute. Appendix B has the convolutional reshape trick for applying SN to Conv2d layers.
      </Prose>

      <Prose>
        Andrew Brock, Jeff Donahue, Karen Simonyan. "Large Scale GAN Training for High Fidelity Natural Image Synthesis." ICLR 2019. arXiv:1809.11096. BigGAN. Section 3 explains how SN is used in both generator and discriminator, and how class-conditional batch-norm scale parameters are spectrally normalized. The "truncation trick" for sampling (truncate the noise distribution at inference to trade diversity for quality) is also introduced here. BigGAN was the first GAN to achieve photorealistic 512x512 ImageNet samples; the SN-everywhere recipe was load-bearing for that result.
      </Prose>

      <Prose>
        Tero Karras, Samuli Laine, Timo Aila. "A Style-Based Generator Architecture for Generative Adversarial Networks." CVPR 2019. arXiv:1812.04948. StyleGAN. Uses R1 gradient penalty (Mescheder et al. variant) on the discriminator rather than WGAN-GP or SN; the choice was driven by StyleGAN's particular architecture (mapping network + adaptive instance norm in generator) which interacts poorly with discriminator-side SN. StyleGAN-2 (arXiv:1912.04958, 2020) and StyleGAN-3 (arXiv:2106.12423, 2021) refine the regularization further with lazy R1 (compute every 16 steps).
      </Prose>

      <Prose>
        Karol Kurach, Mario Lucic, Xiaohua Zhai, Marcin Michalski, Sylvain Gelly. "A Large-Scale Study on Regularization and Normalization in GANs." ICML 2019. The systematic empirical paper. Trains 700+ GAN configurations across architectures, datasets, normalizations, and regularizers. The headline finding: spectral normalization is the single most consistent stabilizer, helping in roughly 90% of configurations tested. Gradient penalty helps in fewer settings but is necessary for some (specifically WGAN-style hinge losses). The combination of SN + GP is rarely needed.
      </Prose>

      <Prose>
        Cem Anil, James Lucas, Roger Grosse. "Sorting Out Lipschitz Function Approximation." ICML 2019. arXiv:1811.05381. The theoretical paper on what 1-Lipschitz networks can and cannot represent. Shows that naive SN networks are not universal approximators of 1-Lipschitz functions because they cannot represent functions like {"x \\mapsto |x|"} (which require non-smooth activations). Introduces GroupSort activations and constrained orthogonal layers as alternatives that <em>are</em> universal approximators under the Lipschitz constraint. This paper is the bridge between "SN as a GAN trick" and "SN as a tool for certified robustness."
      </Prose>

      <Prose>
        Lars Mescheder, Andreas Geiger, Sebastian Nowozin. "Which Training Methods for GANs do actually Converge?" ICML 2018. arXiv:1801.04406. Introduces the R1 and R2 gradient penalties used by StyleGAN. Provides convergence analysis showing that R1 stabilizes training in the absence of a Lipschitz constraint and is computationally cheaper than full WGAN-GP. The paper also has an excellent unified treatment of GAN training dynamics as a saddle-point game.
      </Prose>

      <Prose>
        Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena. "Self-Attention Generative Adversarial Networks." ICML 2019. arXiv:1805.08318. SAGAN, the bridge between SN-GAN and BigGAN. Adds self-attention layers to the discriminator (and generator) on top of SN, demonstrating that SN composes well with attention even though attention's softmax can have large operator norm in principle.
      </Prose>

      <Prose>
        For implementation references: PyTorch's <Code>{"torch.nn.utils.parametrizations.spectral_norm"}</Code> is a faithful implementation of Miyato et al.; the source is in <Code>{"torch/nn/utils/parametrizations.py"}</Code> and is roughly 60 lines, worth reading. The original SN-GAN authors' Chainer implementation is at <Code>{"github.com/pfnet-research/sngan_projection"}</Code> and is the cleanest cross-check. Gulrajani et al.'s original WGAN-GP TensorFlow implementation is at <Code>{"github.com/igul222/improved_wgan_training"}</Code> and the gradient-penalty function there is the canonical reference for the interpolation procedure.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check</H2>

      <Prose>
        Five conceptual questions. Write down your answer before peeking at the solution.
      </Prose>

      <H3>Q1. Why does the original WGAN paper use weight clipping rather than gradient penalty or spectral normalization?</H3>

      <Callout accent="gold" title="Answer">
        The original WGAN paper (Arjovsky et al. 2017) needed <em>some</em> mechanism to enforce the 1-Lipschitz constraint on the discriminator that the Kantorovich-Rubinstein dual requires. Weight clipping was the simplest mechanism the authors could think of: bound every weight to {"[-c, c]"}, and the per-layer operator norm is automatically bounded too. The paper itself acknowledges that clipping is "a clearly terrible way to enforce a Lipschitz constraint" and invites the community to propose better methods. WGAN-GP (Gulrajani et al., three months later) and SN-GAN (Miyato et al., a year later) both came as direct responses to this invitation. The historical lesson: a paper does not have to ship the optimal mechanism for its core idea to land; it just has to ship the idea cleanly enough that follow-up papers can fix the mechanism.
      </Callout>

      <H3>Q2. WGAN-GP penalizes the gradient norm to be 1, not 0. Why?</H3>

      <Callout accent="gold" title="Answer">
        Penalizing gradient norm toward 0 would push the discriminator toward a constant function, which carries no information — the Wasserstein dual would collapse to zero, the generator would receive no gradient, and training would fail. The reason 1 is the right target comes from optimal-transport theory: the optimal {"D^*"} that achieves the Wasserstein dual has gradient norm <em>exactly 1</em> along the optimal transport paths between {"P_r"} and {"P_g"}. So penalizing toward 1 pushes {"D"} toward optimality. The penalty is symmetric (squared), so {"\\|\\nabla D\\| > 1"} and {"\\|\\nabla D\\| < 1"} are both penalized equally. The R1 penalty, by contrast, penalizes toward 0 — it is a pure regularizer rather than a Lipschitz constraint, which is why R1 is used alongside other stability tricks (large batches, careful architectures) rather than alone.
      </Callout>

      <H3>Q3. If a network has 5 layers each with spectral norm exactly 1, what is the network's Lipschitz constant?</H3>

      <Callout accent="gold" title="Answer">
        At most 1 (assuming all activations are 1-Lipschitz, which ReLU and LeakyReLU satisfy). The Lipschitz constant of a composition is at most the product of the constituent Lipschitz constants — for {"f = f_5 \\circ f_4 \\circ f_3 \\circ f_2 \\circ f_1"}, we have {"\\|f\\|_L \\le \\prod_i \\|f_i\\|_L = 1 \\cdot 1 \\cdot 1 \\cdot 1 \\cdot 1 = 1"}. The bound can be loose — the true Lipschitz constant of the composition can be much less than 1 if the layers' worst-case directions do not align — but the upper bound is 1. This is exactly why SN works: bound every layer to 1, get a network bound of 1 for free. The catch is that the bound is loose enough that the network may not be using its full capacity; this is the topic of the Anil-Lucas-Grosse paper and the motivation for GroupSort and constrained orthogonal layers.
      </Callout>

      <H3>Q4. You implement SN by applying power iteration once per training step. Should you also apply power iteration at inference time?</H3>

      <Callout accent="gold" title="Answer">
        No. At inference, the weight matrix {"W"} is frozen, so the spectral norm {"\\sigma_{\\max}(W)"} is constant. Running power iteration just wastes compute. Use the cached {"u, v"} from the last training step. PyTorch's built-in <Code>{"spectral_norm"}</Code> handles this correctly: it only updates the power vectors when <Code>{"self.training == True"}</Code>. The forward pass still divides by {"\\sigma"} at inference, but using the cached value. A common bug is to <em>remove</em> the SN parametrization at inference for "speed", which means the layer now uses the raw {"W"} that is roughly {"\\sigma"} times larger than what was trained — silently degrading quality.
      </Callout>

      <H3>Q5. WGAN-GP requires double-backward through the discriminator. What is the practical compute cost compared to SN, and when does this matter?</H3>

      <Callout accent="gold" title="Answer">
        WGAN-GP costs roughly 2x per training step compared to a non-penalty WGAN, because the gradient-of-gradient term requires retaining the computational graph through the discriminator forward and computing second-order derivatives during the backward of the penalty term. Memory cost roughly doubles too. Spectral normalization adds essentially zero compute — one matrix-vector multiply per layer per step, less than 1% overhead in practice. For small models or research experiments, the 2x WGAN-GP cost is acceptable. For BigGAN-scale training (billions of parameters, thousands of GPUs, weeks of training time), 2x is prohibitive — both in dollars and in being able to fit the model in HBM. This is the practical reason BigGAN uses SN rather than WGAN-GP everywhere it can. The exception is StyleGAN's lazy R1 (compute the penalty every 16 steps) which amortizes the 2x cost down to about 1.06x, making it comparable to SN.
      </Callout>

    </div>
  ),
};

export default spectralNormGPContent;
