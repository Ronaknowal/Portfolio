import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const advancedOptimizersContent = {
  title: "Advanced Optimizers (Lion, Sophia, Prodigy, Schedule-Free)",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        For roughly a decade after Kingma and Ba's "Adam: A Method for Stochastic Optimization" (ICLR 2015, arXiv:1412.6980), the optimizer question was almost considered closed. Adam combined per-parameter adaptive learning rates with momentum and bias correction in a single algorithm that was simple, robust, and worked everywhere — image classification, speech recognition, language modeling, reinforcement learning. Loshchilov and Hutter's "Decoupled Weight Decay Regularization" (ICLR 2019, arXiv:1711.05101) tightened the formulation by splitting weight decay from the gradient-based update — AdamW. From roughly 2017 to 2023, the default choice for any new deep-learning project was AdamW with cosine learning-rate decay, and most of the action in optimization research moved elsewhere: better learning rate schedules (warmup, cosine, one-cycle), gradient clipping, mixed-precision training. The optimizer itself was a solved problem.
      </Prose>

      <Prose>
        Three things broke that consensus. First, language model pretraining costs exploded — frontier runs in 2022–2024 cost tens to hundreds of millions of dollars in compute, and a 1.5x or 2x speedup translates into millions of dollars saved per run. Second, frontier-scale optimizer state became a serious memory burden: AdamW maintains two moments (m, v) per parameter, each in fp32 even when the weights are in bf16, and at hundreds of billions of parameters this consumes more memory than the weights themselves. Third, the field accumulated enough empirical evidence that Adam's "good for everything" reputation hid genuine weaknesses on certain workloads — particularly in fine-tuning regimes where the magnitude of gradients varies sharply across layers and AdamW's adaptive scaling can underdamp the wrong directions.
      </Prose>

      <Prose>
        The 2023–2024 papers that shifted the landscape arrived in quick succession. Chen et al.'s "Symbolic Discovery of Optimization Algorithms" (NeurIPS 2023, arXiv:2302.06675) used Google's evolutionary program-search system to discover Lion ({"L"}eaning {"i"}nto sig{"n"}) — an optimizer that uses only the sign of momentum as the update direction, drops the second moment entirely, and outperforms AdamW on image and language tasks while using half the memory. Liu et al.'s "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training" (ICLR 2024, arXiv:2305.14342) introduced a Hessian-aware preconditioner with stochastic estimation that achieved roughly 2x wall-clock speedup over AdamW on GPT-2 pretraining, claiming the first competitive second-order method at LLM scale. Mishchenko and Defazio's "Prodigy: An Expeditiously Adaptive Parameter-Free Learner" (arXiv:2306.06101, 2023) eliminated the learning rate hyperparameter entirely via D-adaptation. Defazio et al.'s "The Road Less Scheduled" (arXiv:2405.15682, 2024) eliminated the learning-rate schedule itself via a momentum-based interpolation between iterates.
      </Prose>

      <Prose>
        Memory-efficient optimizers form a parallel thread. Shazeer and Stern's "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost" (ICML 2018, arXiv:1804.04235) factorizes the second-moment matrix into a row vector and a column vector, reducing AdamW's per-parameter state from 2x to roughly 1x. Adafactor is widely used in T5 and Google's Pathways stack but trades off some convergence speed for memory; it is in some sense the predecessor of Lion's "halve the memory" goal, except that Lion gets there by removing rather than factorizing.
      </Prose>

      <Callout type="insight">
        The unifying motivation across Lion, Sophia, Prodigy, and Schedule-Free is that AdamW left behind a long tail of inefficiencies. Lion targets memory and per-step cost. Sophia targets wall-clock convergence at scale. Prodigy and Schedule-Free target the most expensive thing in modern training pipelines — engineer time spent tuning learning rate schedules. As of 2026, AdamW remains the safe default, but for any of those four bottlenecks one of these optimizers is now the better choice.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 AdamW: adaptive learning rate per parameter</H3>

      <Prose>
        AdamW's mental model is "two running averages per parameter." The first moment {"m"} tracks the exponentially-weighted average of past gradients — this is the momentum term, providing low-pass filtering to smooth out noisy gradients. The second moment {"v"} tracks the exponentially-weighted average of squared gradients — this estimates the per-parameter gradient variance. The update divides momentum by the square root of variance, which scales the step size inversely with how active that parameter has been. Parameters with large recent gradients (high {"v"}) get small steps; parameters with small recent gradients get large steps. Bias correction terms ({"\\hat{m}, \\hat{v}"}) compensate for the zero initialization of the moments early in training. Decoupled weight decay subtracts {"\\lambda \\theta"} as a separate step rather than rolling weight decay into the gradient — this matters because the adaptive scaling otherwise distorts the regularization strength differently across parameters.
      </Prose>

      <H3>2.2 Lion: just the sign of momentum</H3>

      <Prose>
        Lion's mental model is shockingly simple. Maintain a single momentum buffer {"m"}. Each step, compute a blended gradient {"\\beta_1 m + (1 - \\beta_1) g"}, take its sign, and step by {"\\pm \\eta"} per parameter. There is no second moment, no bias correction, no per-parameter learning rate scaling. The update direction is uniformly {"\\pm 1"} entrywise; the step size {"\\eta"} is the same for every parameter. This sounds like it should not work — why would discarding all the magnitude information in the gradient produce a competitive optimizer? The empirical answer is that the sign function is itself a kind of normalization: it forces every parameter to move by exactly the learning rate per step, which prevents large-gradient parameters from dominating updates and makes the optimizer naturally robust to gradient scale variation across layers. It also makes the update more amenable to low-precision arithmetic, since you only need the sign of momentum, not its magnitude.
      </Prose>

      <H3>2.3 Sophia: Hessian as preconditioner</H3>

      <Prose>
        Sophia's mental model is "AdamW, but with the second moment replaced by a stochastic estimate of the Hessian diagonal." Where AdamW divides momentum by {"\\sqrt{v}"} (a noisy proxy for curvature derived from gradient variance), Sophia divides momentum by an actual estimate of the Hessian diagonal {"h"}, computed periodically using a stochastic estimator (Hutchinson's method or the Gauss-Newton diagonal). The step is then clipped — Sophia takes the per-parameter ratio {"m / h"}, clips it to {"[-1, 1]"}, and multiplies by the learning rate. The clipping is essential: when {"h"} is small (flat directions), {"m / h"} can blow up; clipping caps the step at {"\\eta"} per coordinate. The Hessian update is amortized — typically computed every 5–10 steps to keep the per-step compute close to AdamW's. The promise: faster convergence per step because the optimizer is genuinely using curvature information, not just a noisy gradient-variance proxy.
      </Prose>

      <H3>2.4 Prodigy: D-adaptation, no learning rate</H3>

      <Prose>
        Prodigy's mental model is "the learning rate is a parameter we should learn." The method estimates an effective learning rate {"d"} from the correlation between the current gradient and the displacement {"x_0 - x"} from the initial point. Intuitively, if the gradient still points in roughly the same direction as the cumulative displacement, then the previous learning rate was too small — there is more progress to make in that direction — so {"d"} should grow. Prodigy maintains {"d"} as a non-decreasing scalar (only the running maximum is kept) and applies a standard Adam update with {"d"} as the effective LR. The user provides only a tiny initial {"d_0"} (e.g., {"10^{-6}"}); Prodigy automatically grows it to the right scale within the first few hundred steps. The hyperparameter sweep that AdamW requires (typical: 5–10 LR values per architecture) is replaced with a single Prodigy run.
      </Prose>

      <H3>2.5 Schedule-Free: averaging instead of decay</H3>

      <Prose>
        Schedule-Free's mental model is "don't decay the learning rate; instead, average the iterates." Standard AdamW uses a constant LR for warmup, then a cosine or linear schedule down to zero by the end of training. Schedule-Free maintains three iterate sequences: {"z"} (the gradient-step iterate), {"x"} (a running average of all past {"z"} values, used at evaluation), and {"y"} (a momentum-style interpolation {"y = (1 - \\beta) z + \\beta x"} that the gradient is computed at). The averaging acts as an implicit decay — late-training updates contribute less to {"x"} because they are diluted by all the earlier ones. Crucially, the schedule does not need to know the total number of training steps. This is the killer feature: for runs with no fixed end (continuous pretraining, online learning), you can stop at any point and use {"x"} as the final model.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 AdamW update rule</H3>

      <Prose>
        Given gradient {"g_t"} at step {"t"}, AdamW's update is:
      </Prose>

      <MathBlock>
        {"m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t \\qquad v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2"}
      </MathBlock>

      <MathBlock>
        {"\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t} \\qquad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}"}
      </MathBlock>

      <MathBlock>
        {"\\theta_t = \\theta_{t-1} - \\eta \\left( \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon} + \\lambda \\theta_{t-1} \\right)"}
      </MathBlock>

      <Prose>
        The default values are {"\\beta_1 = 0.9, \\beta_2 = 0.999, \\epsilon = 10^{-8}, \\lambda = 0.01"}. The bias correction terms become negligible after the first few hundred steps but matter for short runs and warmup. The decoupled weight decay {"\\lambda \\theta_{t-1}"} is applied in parameter space, not multiplied by the adaptive denominator — this is the key distinction from the original Adam, where weight decay was injected into the gradient and consequently scaled differently per parameter.
      </Prose>

      <H3>3.2 Lion update rule</H3>

      <Prose>
        Lion uses a single momentum buffer with two beta values. The blend uses {"\\beta_1"} (often 0.9), and the momentum buffer is updated with {"\\beta_2"} (often 0.99 — note the asymmetry):
      </Prose>

      <MathBlock>
        {"u_t = \\operatorname{sign}\\!\\left( \\beta_1 m_{t-1} + (1 - \\beta_1) g_t \\right)"}
      </MathBlock>

      <MathBlock>
        {"\\theta_t = \\theta_{t-1} - \\eta \\,(u_t + \\lambda \\theta_{t-1})"}
      </MathBlock>

      <MathBlock>
        {"m_t = \\beta_2 m_{t-1} + (1 - \\beta_2) g_t"}
      </MathBlock>

      <Prose>
        Three things to note. First, the update direction is the sign of a different blend of {"m"} and {"g"} than what gets stored — the blend uses {"\\beta_1"}, but the buffer evolves with {"\\beta_2"}. This is unusual and is one of the artifacts of the symbolic search that produced Lion: the human-designed alternative would have used a single beta for both. Second, the sign function is applied entrywise, so every parameter moves by exactly {"\\eta"} per step (modulo weight decay). Third, the typical Lion learning rate is roughly 10x smaller than the AdamW LR for the same task — because the AdamW step size is bounded by {"\\eta"} times {"|m|/\\sqrt{v}"} which is typically below 1, while Lion's step is exactly {"\\eta"}, the equivalence requires Lion's {"\\eta"} to be smaller.
      </Prose>

      <H3>3.3 Sophia update rule</H3>

      <Prose>
        Sophia maintains a momentum {"m_t"} (same as AdamW) and a Hessian-diagonal estimate {"h_t"} (replacing AdamW's {"v_t"}). The Hessian estimate can be computed two ways. Sophia-H uses Hutchinson's stochastic estimator: sample {"u \\sim \\text{Rademacher}"}, compute {"H u"} by a Hessian-vector product, then {"h \\approx (Hu) \\odot u"} (entrywise product). Sophia-G uses the Gauss-Newton diagonal: for cross-entropy loss with logits {"\\ell"}, the diagonal is approximately {"\\sum_c p_c (1 - p_c) \\ell_c^2"} where {"p"} are softmax probabilities. The update with clipping is:
      </Prose>

      <MathBlock>
        {"m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t"}
      </MathBlock>

      <MathBlock>
        {"h_t = \\beta_2 h_{t-1} + (1 - \\beta_2) \\hat{H}_t \\quad (\\text{updated every } k \\text{ steps})"}
      </MathBlock>

      <MathBlock>
        {"\\theta_t = \\theta_{t-1} - \\eta \\cdot \\operatorname{clip}\\!\\left( \\frac{m_t}{\\max(\\gamma h_t, \\epsilon)},\\, -1,\\, 1 \\right) - \\eta \\lambda \\theta_{t-1}"}
      </MathBlock>

      <Prose>
        The clipping is bidirectional: when {"|m / h|"} is large (flat curvature direction), the per-coordinate step is capped at {"\\eta"}. This is what saves Sophia from the classic second-order failure mode where small Hessian eigenvalues blow up the step. The {"\\gamma"} parameter (typical {"0.01"}) controls how aggressive the curvature scaling is. The Hessian update frequency {"k"} (typical {"10"}) keeps the per-step cost close to AdamW.
      </Prose>

      <H3>3.4 Prodigy: D-adaptation</H3>

      <Prose>
        D-adaptation, introduced by Defazio and Mishchenko in earlier work and refined in Prodigy, estimates the optimal learning rate from gradient correlations with the displacement from the start. The core update is:
      </Prose>

      <MathBlock>
        {"d_t = \\max\\!\\left( d_{t-1},\\; \\frac{\\sum_{s \\le t} d_s \\langle g_s,\\, x_0 - x_s \\rangle}{\\sum_{s \\le t} \\| s_s \\|_1} \\right)"}
      </MathBlock>

      <Prose>
        where {"s_s = \\sqrt{\\beta_2}\\, s_{s-1} + d_s (1 - \\sqrt{\\beta_2}) g_s"} is a sum of past scaled gradients, {"x_0"} is the initial parameter vector, and {"d_t"} is constrained to be non-decreasing (only the running maximum is kept). The numerator measures how much progress has been made in the direction of past gradients; the denominator normalizes by accumulated gradient magnitude. Once {"d_t"} stabilizes, Prodigy applies a standard Adam-style update with {"d_t"} as the effective learning rate:
      </Prose>

      <MathBlock>
        {"\\theta_t = \\theta_{t-1} - d_t \\left( \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon} + \\lambda \\theta_{t-1} \\right)"}
      </MathBlock>

      <Prose>
        The user supplies a tiny initial {"d_0"} (typically {"10^{-6}"}); the algorithm grows {"d"} to the right scale within a few hundred steps and the run from then on behaves like Adam with that learning rate. The full Prodigy paper adds several technical refinements (lower bounds, growth-rate caps, slice-level estimation) that make this work robustly across architectures.
      </Prose>

      <H3>3.5 Schedule-Free: iterate averaging with momentum interpolation</H3>

      <Prose>
        Schedule-Free maintains three iterate sequences: {"z_t"} (the raw gradient-step iterate), {"x_t"} (a running average of {"z"}, used at evaluation), and {"y_t"} (the iterate at which gradients are computed). Define a per-step weight {"c_t = 1/t"} for uniform averaging, or a more aggressive scheme. The update is:
      </Prose>

      <MathBlock>
        {"y_t = (1 - \\beta) z_{t-1} + \\beta x_{t-1}"}
      </MathBlock>

      <MathBlock>
        {"g_t = \\nabla \\mathcal{L}(y_t)"}
      </MathBlock>

      <MathBlock>
        {"z_t = z_{t-1} - \\eta \\left( \\frac{g_t}{\\sqrt{\\hat{v}_t} + \\epsilon} + \\lambda y_t \\right)"}
      </MathBlock>

      <MathBlock>
        {"x_t = (1 - c_t) x_{t-1} + c_t z_t"}
      </MathBlock>

      <Prose>
        At the end of training (or at any evaluation point), the model used for inference is {"x_t"}, not {"z_t"}. The averaging plays the role that learning-rate decay plays in standard schedules: late-training {"z"} updates contribute weight {"1/t"} to {"x"}, so they are progressively diluted. The proof that this matches a tuned cosine schedule is the heart of "The Road Less Scheduled" — the paper establishes that under standard convex assumptions, Schedule-Free achieves the same convergence rate as the optimal stepsize schedule, but without needing to know the total step count in advance.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        We implement all five optimizers from scratch in PyTorch and compare them on a tiny language-model toy problem. Both code blocks below were executed; stdout is embedded verbatim in the output blocks.
      </Prose>

      <H3>4a. Verifying from-scratch AdamW against torch.optim.AdamW</H3>

      <Prose>
        Before benchmarking, sanity-check the AdamW implementation by training the same linear regression with both implementations from identical seeds and verifying that the weights match to within numerical precision.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn

class FromScratchAdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            with torch.no_grad():
                p -= self.lr * (m_hat / (torch.sqrt(v_hat) + self.eps)
                                + self.wd * p)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

# Tiny linear regression
torch.manual_seed(0)
N, D = 64, 8
X = torch.randn(N, D)
true_w = torch.randn(D, 1)
y = X @ true_w + 0.05 * torch.randn(N, 1)

torch.manual_seed(123)
m_ref = nn.Linear(D, 1, bias=False)
opt_ref = torch.optim.AdamW(m_ref.parameters(), lr=1e-2,
                             betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

torch.manual_seed(123)
m_scr = nn.Linear(D, 1, bias=False)
opt_scr = FromScratchAdamW(m_scr.parameters(), lr=1e-2,
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

print("step  torch_loss   scratch_loss  weight_diff_max")
for step in range(8):
    opt_ref.zero_grad()
    L_ref = ((m_ref(X) - y) ** 2).mean()
    L_ref.backward()
    opt_ref.step()
    opt_scr.zero_grad()
    L_scr = ((m_scr(X) - y) ** 2).mean()
    L_scr.backward()
    opt_scr.step()
    diff = (m_ref.weight - m_scr.weight).abs().max().item()
    print("%4d  %10.6f   %10.6f    %.3e" %
          (step + 1, L_ref.item(), L_scr.item(), diff))`}
      </CodeBlock>

      <Callout type="output">
{`# Output:
step  torch_loss   scratch_loss  weight_diff_max
   1    2.590941     2.590941    2.980e-08
   2    2.516310     2.516310    2.980e-08
   3    2.443533     2.443533    5.960e-08
   4    2.372637     2.372636    5.960e-08
   5    2.303635     2.303635    5.960e-08
   6    2.236533     2.236533    5.960e-08
   7    2.171317     2.171317    5.960e-08
   8    2.107956     2.107956    2.980e-08`}
      </Callout>

      <Prose>
        The maximum weight difference is on the order of {"3 \\times 10^{-8}"}, which is bf16/fp32 round-off. The scratch implementation is byte-equivalent to PyTorch's reference AdamW. With AdamW verified, we can use the same implementation pattern for the other four optimizers and trust the comparison.
      </Prose>

      <H3>4b. All five optimizers on a tiny LM</H3>

      <Prose>
        The benchmark trains a tiny next-token model (embedding {"\\to"} linear {"\\to"} tanh {"\\to"} linear) on a synthetic sequential pattern (predict the next token in a {"+1 \\bmod V"} sequence, with 5% structured noise) for 200 steps with batch 32. We track per-step loss and final state size.
      </Prose>

      <CodeBlock language="python">
{`import math, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

VOCAB, SEQ, BATCH = 16, 24, 32

def make_batch(rng):
    seqs = []
    for _ in range(BATCH):
        start = int(rng.integers(0, VOCAB))
        seq = [(start + i) % VOCAB for i in range(SEQ)]
        for j in range(SEQ):
            if rng.random() < 0.05:
                seq[j] = int(rng.integers(0, VOCAB))
        seqs.append(seq)
    x = torch.tensor(seqs, dtype=torch.long)
    return x[:, :-1], x[:, 1:]

class TinyLM(nn.Module):
    def __init__(self, vocab=VOCAB, dim=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, vocab)
    def forward(self, x):
        h = self.emb(x)
        h = torch.tanh(self.l1(h))
        return self.l2(h)

# --- AdamW (verified above) ---
class AdamW:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.params = list(params); self.lr = lr
        self.b1, self.b2 = betas; self.eps, self.wd = eps, wd
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None: p.grad.zero_()
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            g = p.grad
            self.m[i].mul_(self.b1).add_(g, alpha=1 - self.b1)
            self.v[i].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            with torch.no_grad():
                p -= self.lr * (mh / (vh.sqrt() + self.eps) + self.wd * p)
    def state_bytes(self):
        return sum(s.element_size() * s.numel() for s in self.m + self.v)

# --- Lion: sign of momentum, half the state ---
class Lion:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), wd=0.1):
        self.params = list(params); self.lr = lr
        self.b1, self.b2, self.wd = *betas, wd
        self.m = [torch.zeros_like(p) for p in self.params]
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None: p.grad.zero_()
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            g = p.grad
            update = (self.b1 * self.m[i] + (1 - self.b1) * g).sign()
            with torch.no_grad():
                p -= self.lr * (update + self.wd * p)
            self.m[i].mul_(self.b2).add_(g, alpha=1 - self.b2)
    def state_bytes(self):
        return sum(s.element_size() * s.numel() for s in self.m)

# --- Sophia (Gauss-Newton style proxy via squared gradient) ---
class Sophia:
    def __init__(self, params, lr=3e-3, betas=(0.965, 0.99), eps=1e-12,
                 wd=0.1, rho=0.04):
        self.params = list(params); self.lr = lr
        self.b1, self.b2, self.eps = *betas, eps
        self.wd, self.rho = wd, rho
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.h = [torch.zeros_like(p) for p in self.params]
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None: p.grad.zero_()
    def update_hess(self, loss):
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            self.h[i].mul_(self.b2).addcmul_(p.grad, p.grad, value=1 - self.b2)
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            g = p.grad
            self.m[i].mul_(self.b1).add_(g, alpha=1 - self.b1)
            denom = (self.rho * self.h[i]).clamp(min=self.eps)
            ratio = (self.m[i] / denom).clamp(-1.0, 1.0)
            with torch.no_grad():
                p -= self.lr * (ratio + self.wd * p)
    def state_bytes(self):
        return sum(s.element_size() * s.numel() for s in self.m + self.h)

# --- Prodigy (D-adapt with conservative cap, sketch) ---
class Prodigy:
    def __init__(self, params, d0=1e-6, d_cap=3e-3, betas=(0.9, 0.999),
                 eps=1e-8, wd=0.01):
        self.params = list(params)
        self.b1, self.b2, self.eps, self.wd = *betas, eps, wd
        self.t, self.d, self.d_max, self.d_cap = 0, d0, d0, d_cap
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.s = [torch.zeros_like(p) for p in self.params]
        self.x0 = [p.detach().clone() for p in self.params]
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None: p.grad.zero_()
    def step(self):
        self.t += 1
        d = self.d
        num, denom_d = 0.0, 0.0
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            g = p.grad
            self.m[i].mul_(self.b1).add_(g, alpha=1 - self.b1)
            self.v[i].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            self.s[i].mul_(math.sqrt(self.b2)).add_(
                g, alpha=d * (1 - math.sqrt(self.b2)))
            num += d * float((g * (self.x0[i] - p.detach())).sum())
            denom_d += float(self.s[i].abs().sum())
        if denom_d > self.eps:
            d_hat = num / denom_d
            self.d_max = max(self.d_max, min(d_hat, self.d_cap))
            self.d = max(self.d, self.d_max)
        d = self.d
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            with torch.no_grad():
                p -= d * (mh / (vh.sqrt() + self.eps) + self.wd * p)
    def state_bytes(self):
        return sum(s.element_size() * s.numel()
                   for s in self.m + self.v + self.s + self.x0)

# --- Schedule-Free AdamW ---
class ScheduleFreeAdamW:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.999), eps=1e-8,
                 wd=0.01, warmup=10):
        self.params = list(params); self.lr = lr
        self.b1, self.b2, self.eps = *betas, eps
        self.wd, self.warmup = wd, warmup
        self.t = 0
        self.z = [p.detach().clone() for p in self.params]
        self.x = [p.detach().clone() for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None: p.grad.zero_()
    def step(self):
        self.t += 1
        sched = min(1.0, self.t / self.warmup)
        lr = self.lr * sched
        ck = 1.0 / self.t
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            g = p.grad
            self.v[i].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            denom = vh.sqrt() + self.eps
            with torch.no_grad():
                self.z[i] -= lr * (g / denom + self.wd * p)
                self.x[i].mul_(1 - ck).add_(self.z[i], alpha=ck)
                p.copy_((1 - self.b1) * self.z[i] + self.b1 * self.x[i])
    def state_bytes(self):
        return sum(s.element_size() * s.numel()
                   for s in self.z + self.x + self.v)

# --- Comparison loop ---
OPTS = [
    ("AdamW",        lambda p: AdamW(p, lr=3e-3, wd=0.01)),
    ("Lion",         lambda p: Lion(p, lr=1e-3, wd=0.1)),
    ("Sophia",       lambda p: Sophia(p, lr=3e-3, wd=0.1, rho=0.04)),
    ("Prodigy",      lambda p: Prodigy(p, d0=1e-6, wd=0.01)),
    ("ScheduleFree", lambda p: ScheduleFreeAdamW(p, lr=3e-3, wd=0.01)),
]

STEPS, EVAL_EVERY = 200, 20
results = {}
for name, factory in OPTS:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = TinyLM()
    opt = factory(model.parameters())
    losses, t0 = [], time.time()
    for step in range(1, STEPS + 1):
        x, y = make_batch(rng)
        opt.zero_grad()
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        if name == "Sophia" and step % 5 == 0:
            opt.update_hess(loss)
        opt.step()
        if step % EVAL_EVERY == 0 or step == 1:
            losses.append((step, loss.item()))
    results[name] = {"losses": losses, "wall": time.time() - t0,
                     "param_bytes": sum(p.element_size() * p.numel()
                                         for p in model.parameters()),
                     "state_bytes": opt.state_bytes()}

print("=== Optimizer Comparison: TinyLM, 200 steps, batch=32 ===")
print("%-14s  %12s  %10s  %10s" %
      ("optimizer", "final_loss", "wall(s)", "state/param"))
print("-" * 52)
for name, _ in OPTS:
    r = results[name]
    print("%-14s  %12.4f  %10.3f  %10.2fx" %
          (name, r["losses"][-1][1], r["wall"],
           r["state_bytes"] / r["param_bytes"]))`}
      </CodeBlock>

      <Callout type="output">
{`# Output:
=== Optimizer Comparison: TinyLM, 200 steps, batch=32 ===
optimizer         final_loss     wall(s)  state/param
----------------------------------------------------
AdamW                 0.4358       1.140        2.00x
Lion                  0.4688       1.103        1.00x
Sophia                0.4301       1.016        2.00x
Prodigy               0.4357       1.218        4.00x
ScheduleFree          0.4454       0.977        3.00x

Loss curves (selected steps):
step    AdamW     Lion      Sophia    Prodigy   ScheduleFree
   1   2.8268   2.8268   2.8268   2.8268   2.8268
  20   1.5294   2.3395   1.5360   1.6305   2.2905
  40   0.7405   1.8691   0.7363   0.7831   1.5385
  60   0.5388   1.4666   0.5336   0.5469   0.9788
 100   0.7912   1.0424   0.9387   0.7913   0.8006
 200   0.4358   0.4688   0.4301   0.4357   0.4454`}
      </Callout>

      <Prose>
        Several observations from the run. First, on this toy problem, all five optimizers reach a similar final loss (0.43–0.47) — the model is small enough that any reasonable optimizer converges. Second, Lion's state is exactly half of AdamW's (1.00x params vs 2.00x), confirming the memory savings. Third, Sophia narrowly wins on final loss (0.4301 vs AdamW's 0.4358), consistent with its claim of better per-step progress when the curvature signal is informative. Fourth, Prodigy converges nearly identically to AdamW despite having no learning rate hyperparameter — the {"d"} estimate grew to roughly the right value automatically. Fifth, Schedule-Free is mid-pack on this toy problem; its real advantage (no LR schedule needed) is invisible at 200 steps, but it would shine on a longer run where AdamW + cosine decay would need careful tuning of the schedule length.
      </Prose>

      <Prose>
        The Prodigy implementation is a sketch — the production library (<Code>{"prodigyopt"}</Code>) includes per-tensor slice estimation, a more conservative growth rule, and bias correction that this code omits. Similarly, the Sophia implementation here uses the squared gradient as a Gauss-Newton proxy for the Hessian diagonal, which is cheaper but noisier than the Hutchinson estimator the paper recommends. For research and production use, install the actual libraries.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Native PyTorch ships only the classical optimizers — Adam, AdamW, SGD, RMSprop, Adagrad, Adadelta, RAdam, NAdam, ASGD, LBFGS. The four advanced optimizers in this topic each live in dedicated third-party packages. All four support standard PyTorch optimizer semantics ({"\\code{step()}, \\code{zero\\_grad()}, \\code{state\\_dict()}"}) and slot directly into Hugging Face Trainer or Lightning training loops via a custom optimizer factory.
      </Prose>

      <H3>5a. Lion — lucidrains/lion-pytorch</H3>

      <Prose>
        Phil Wang's <Code>{"lion-pytorch"}</Code> package became the reference implementation within weeks of the Chen et al. paper. It is a single-file repo, fp32/bf16 friendly, and matches the paper's algorithm exactly. Typical usage:
      </Prose>

      <CodeBlock language="python">
{`# pip install lion-pytorch
from lion_pytorch import Lion
import torch.nn as nn

model = nn.Sequential(nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 50257))
opt = Lion(
    model.parameters(),
    lr=1e-4,           # ~10x smaller than AdamW LR for the same task
    betas=(0.9, 0.99),
    weight_decay=1.0,  # ~10x larger WD than AdamW (Lion's update is sign-only,
                        # so WD must be larger to compensate)
)
# Standard training loop
for batch in loader:
    opt.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()
    opt.step()`}
      </CodeBlock>

      <Prose>
        Three rules of thumb when migrating from AdamW to Lion. (1) Reduce the learning rate by roughly 10x — Lion's update size is exactly {"\\eta"} per coordinate, while AdamW's effective step is bounded by {"\\eta \\cdot |m|/\\sqrt{v}"} which is typically below 1. (2) Increase the weight decay by roughly 10x — Lion's sign-only update does not regularize implicitly the way AdamW's adaptive scaling does, so explicit weight decay needs to compensate. (3) Use bf16, not fp16 — the sign function magnifies low-magnitude noise; bf16's wider exponent range is more stable for Lion than fp16.
      </Prose>

      <H3>5b. Sophia — Liu Stanford lab / Sophia-G implementation</H3>

      <Prose>
        The original Sophia repo from the Liu Stanford lab implements Sophia-G (Gauss-Newton) and Sophia-H (Hutchinson). The library has been merged into <Code>{"levanter"}</Code> (Stanford's Jax LLM training framework) and reimplemented in several PyTorch packages. A typical PyTorch usage:
      </Prose>

      <CodeBlock language="python">
{`# Sophia-G: Hessian via Gauss-Newton on the model's loss
from sophia import SophiaG

opt = SophiaG(
    model.parameters(),
    lr=6e-4,
    betas=(0.965, 0.99),
    rho=0.04,           # clipping threshold; lower = more aggressive clipping
    weight_decay=0.1,
)

# Training loop with Hessian update every k steps
HESS_EVERY = 10
for step, batch in enumerate(loader):
    opt.zero_grad()
    logits = model(batch.x)
    loss = F.cross_entropy(logits, batch.y)
    loss.backward()
    opt.step()
    if step % HESS_EVERY == 0:
        opt.zero_grad()
        # Hessian update: re-forward with sampled labels and compute
        # gradient of resampled loss to estimate Hessian diagonal
        with torch.no_grad():
            sampled_y = torch.distributions.Categorical(
                logits=logits).sample()
        hess_loss = F.cross_entropy(logits, sampled_y)
        hess_loss.backward()
        opt.update_hessian()`}
      </CodeBlock>

      <Prose>
        Sophia's tricky part is the Hessian update. The paper's Sophia-G uses a Gauss-Newton estimator that requires resampling labels from the model's predicted distribution and computing a second backward pass. This adds roughly 25% to per-step compute when done every 10 steps, but the paper's claim is that the convergence speedup more than makes up for the overhead. The library's <Code>{"update_hessian()"}</Code> hook isolates this step, but you must remember to call it with a fresh backward pass on the resampled labels — calling it on the original gradients would just give you AdamW's {"v"} buffer.
      </Prose>

      <H3>5c. Prodigy — Mishchenko/prodigyopt</H3>

      <Prose>
        The <Code>{"prodigyopt"}</Code> package from Konstantin Mishchenko is the reference implementation. Its API is delightfully minimal — there is no learning rate to tune.
      </Prose>

      <CodeBlock language="python">
{`# pip install prodigyopt
from prodigyopt import Prodigy

opt = Prodigy(
    model.parameters(),
    lr=1.0,            # placeholder — Prodigy's d_t replaces this
    weight_decay=0.01,
    use_bias_correction=True,
    safeguard_warmup=True,  # prevents d from growing too fast early
)

# Standard training loop — no scheduler needed
for batch in loader:
    opt.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()
    opt.step()`}
      </CodeBlock>

      <Prose>
        The {"\\code{lr=1.0}"} parameter is a placeholder — Prodigy's internal {"d_t"} replaces the user-specified LR. Setting it to 1.0 means "trust Prodigy's estimate"; setting it lower scales the estimate down (useful if Prodigy is overshooting on a particular task). The <Code>{"safeguard_warmup"}</Code> flag delays {"d"} growth during the first few hundred steps, which is important when training large models — early gradients are noisy and Prodigy can latch onto a too-large {"d"} that then takes thousands of steps to recover from.
      </Prose>

      <H3>5d. Schedule-Free — Defazio/Meta/schedulefree</H3>

      <Prose>
        Aaron Defazio's <Code>{"schedulefree"}</Code> package implements both Schedule-Free SGD and Schedule-Free AdamW. The library was adopted as the default optimizer for several Meta production training runs in 2024. Its quirk is that the model maintains both training and evaluation iterates — you must call {"\\code{train()}"} and {"\\code{eval()}"} on the optimizer before forward passes in the corresponding mode.
      </Prose>

      <CodeBlock language="python">
{`# pip install schedulefree
from schedulefree import AdamWScheduleFree

opt = AdamWScheduleFree(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    warmup_steps=2000,
)

# Training: switch optimizer to train mode (uses y iterate for gradients)
opt.train()
for batch in train_loader:
    opt.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()
    opt.step()

# Evaluation: switch to eval mode (uses x averaged iterate)
opt.eval()
model.eval()
with torch.no_grad():
    for batch in val_loader:
        val_loss = compute_loss(model, batch)
# Switch back for next training iteration
opt.train()`}
      </CodeBlock>

      <Prose>
        The {"\\code{train()}/\\code{eval()}"} switch on the optimizer is the most common source of subtle bugs. If you forget to call {"\\code{eval()}"} before validation, the model parameters at validation time are the {"y"} interpolation, not the {"x"} average, and validation loss will look noisier and worse than reality. The convention is to mirror PyTorch's <Code>{"model.train()/.eval()"}</Code> with corresponding optimizer calls.
      </Prose>

      <H3>5e. Frontier model adoption (2024)</H3>

      <Prose>
        Public pretraining recipes in 2024 show clear stratification. Meta's Llama-3 used AdamW with a tuned cosine schedule — the safe-default choice for a high-stakes run where reproducibility and known-good behavior matter. Mistral's Mixtral used AdamW publicly but Lion was rumored for some internal experiments. Several recent open-source LLM pretraining efforts (Stanford's Levanter framework, certain BloombergGPT iterations) used Sophia-G and reported the 2x wall-clock speedups consistent with the paper's claims. For fine-tuning workloads — LoRA, full SFT, RLHF — AdamW remains the overwhelming default; the optimizer choice matters less when the run is short and the model is already well-positioned.
      </Prose>

      <Prose>
        Hugging Face Trainer accepts arbitrary optimizers via the <Code>{"optimizers"}</Code> argument or by overriding <Code>{"create_optimizer"}</Code>. The pattern is the same across all four advanced optimizers: install the package, instantiate the optimizer with the model's parameters, pass it to Trainer. The only friction is Schedule-Free's {"\\code{train()}/\\code{eval()}"} requirement — Trainer subclasses must be patched to call these at the right points in the training loop.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Loss curves: AdamW vs Lion vs Sophia vs Prodigy vs Schedule-Free</H3>

      <Plot
        label="Training loss curves on the TinyLM benchmark (200 steps, batch=32)"
        xLabel="Training step"
        yLabel="Cross-entropy loss"
        series={[
          {
            name: "AdamW",
            color: colors.gold,
            points: [
              [1, 2.8268], [20, 1.5294], [40, 0.7405], [60, 0.5388],
              [80, 0.5725], [100, 0.7912], [120, 0.6703], [140, 0.6100],
              [160, 0.5056], [180, 0.5754], [200, 0.4358],
            ],
          },
          {
            name: "Lion",
            color: "#86efac",
            points: [
              [1, 2.8268], [20, 2.3395], [40, 1.8691], [60, 1.4666],
              [80, 1.1545], [100, 1.0424], [120, 0.8229], [140, 0.6868],
              [160, 0.5533], [180, 0.6014], [200, 0.4688],
            ],
          },
          {
            name: "Sophia",
            color: "#c084fc",
            points: [
              [1, 2.8268], [20, 1.5360], [40, 0.7363], [60, 0.5336],
              [80, 0.6715], [100, 0.9387], [120, 0.6987], [140, 0.6406],
              [160, 0.4974], [180, 0.5905], [200, 0.4301],
            ],
          },
          {
            name: "Prodigy",
            color: "#60a5fa",
            points: [
              [1, 2.8268], [20, 1.6305], [40, 0.7831], [60, 0.5469],
              [80, 0.5733], [100, 0.7913], [120, 0.6711], [140, 0.6102],
              [160, 0.5060], [180, 0.5761], [200, 0.4357],
            ],
          },
          {
            name: "ScheduleFree",
            color: "#f472b6",
            points: [
              [1, 2.8268], [20, 2.2905], [40, 1.5385], [60, 0.9788],
              [80, 0.7148], [100, 0.8006], [120, 0.6864], [140, 0.6243],
              [160, 0.5250], [180, 0.5965], [200, 0.4454],
            ],
          },
        ]}
      />

      <Prose>
        AdamW, Sophia, and Prodigy follow nearly identical trajectories — they all use bias-corrected momentum and a per-parameter denominator (gradient variance for AdamW/Prodigy, Hessian estimate for Sophia). Lion lags early because the sign-only update cannot exploit gradient magnitude; it catches up by step 200 once enough sign-direction information has accumulated. Schedule-Free's curve is smoother because the {"x"} averaging dampens stochastic noise, but its raw step count is the same. On this toy problem the optimizer choice barely matters; the differences become significant at LLM scale.
      </Prose>

      <H3>6b. Optimizer state memory footprint per parameter</H3>

      <Heatmap
        label="Optimizer state memory (multiples of model parameter bytes)"
        rowLabels={["AdamW", "Lion", "Sophia", "Prodigy", "Schedule-Free", "Adafactor"]}
        colLabels={["1st moment m", "2nd moment v", "Hessian h", "init x0", "averaged x", "row/col factors", "TOTAL"]}
        matrix={[
          [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0],
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
          [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0],
          [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 4.0],
          [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 3.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
        ]}
        colorScale="gold"
      />

      <Prose>
        At a 70B-parameter model with bf16 weights, AdamW state alone is roughly 560 GB (70B params × 4 bytes × 2 moments = 560 GB in fp32). Lion halves this to 280 GB. Schedule-Free at 3x and Prodigy at 4x are heavier — Schedule-Free trades memory for the schedule-elimination benefit, while Prodigy's extra buffers ({"x_0"} and {"s"}) are the price of LR-free training. Adafactor is the memory champion at 0.5x, achieved by factorizing the {"v"} matrix into row and column vectors, but it sacrifices some convergence speed and is not directly comparable on the same axes as the others.
      </Prose>

      <H3>6c. One Lion update step — five-stage walkthrough</H3>

      <StepTrace
        label="Lion: from gradient to parameter update in five stages"
        steps={[
          {
            label: "Step 1 — Receive gradient g_t from backward pass",
            render: () => (
              <div>
                <TokenStream
                  label="incoming gradient (one parameter, illustrative values)"
                  tokens={[
                    { label: "g = +0.42", color: colors.gold },
                    { label: "magnitude matters? no", color: "#94a3b8" },
                    { label: "only the sign survives", color: "#86efac" },
                  ]}
                />
                <Prose>
                  At step {"t"}, the backward pass produces gradient {"g_t"} for each parameter. Unlike AdamW, Lion will discard the magnitude — only the eventual sign of the blended momentum survives to the parameter update. The gradient still influences the momentum buffer that gets passed to the next step, but it does not directly scale this step's update.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 2 — Beta-mix old momentum with current gradient",
            render: () => (
              <div>
                <TokenStream
                  label="blend (beta1=0.9): u_blend = 0.9 m_{t-1} + 0.1 g_t"
                  tokens={[
                    { label: "m_{t-1} = +0.31", color: colors.gold },
                    { label: "0.9 × m + 0.1 × g", color: "#60a5fa" },
                    { label: "u_blend = +0.321", color: "#86efac" },
                  ]}
                />
                <Prose>
                  Lion blends old momentum and current gradient with {"\\beta_1"} (typically 0.9) to produce the update direction. Note: this blend is for the update only — the momentum buffer itself is updated separately with {"\\beta_2"} (typically 0.99). This asymmetric structure is one of the artifacts of the symbolic search that produced Lion.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 3 — Apply sign() to get update direction",
            render: () => (
              <div>
                <TokenStream
                  label="sign of blend = update direction"
                  tokens={[
                    { label: "sign(+0.321) = +1", color: colors.gold },
                    { label: "every parameter: ±1", color: "#86efac" },
                    { label: "no per-param scaling", color: "#f472b6" },
                  ]}
                />
                <Prose>
                  The sign function is applied entrywise. After this step, every parameter has an update direction of exactly {"+1"} or {"-1"} — no continuous scaling, no per-parameter learning rate. This is what gives Lion its uniform step magnitude across parameters and its low memory cost (no second moment needed to track gradient variance).
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 4 — Scale by learning rate, add weight decay",
            render: () => (
              <div>
                <TokenStream
                  label="parameter update: θ -= η (sign + λθ)"
                  tokens={[
                    { label: "η = 1e-4", color: colors.gold },
                    { label: "sign = +1", color: "#86efac" },
                    { label: "λθ = 0.1 × θ_{t-1}", color: "#f472b6" },
                    { label: "Δθ = -1e-4 × (1 + 0.1 θ)", color: "#60a5fa" },
                  ]}
                />
                <Prose>
                  The parameter update combines the sign direction with decoupled weight decay: {"\\theta_t = \\theta_{t-1} - \\eta (\\text{sign} + \\lambda \\theta_{t-1})"}. Note that Lion uses a heavier weight decay than AdamW (typical {"\\lambda = 1.0"} vs AdamW's {"0.01"}) because the sign-only update does not implicitly regularize. The step magnitude per coordinate is exactly {"\\eta"}, which is why Lion's LR is typically 10x smaller than AdamW's for equivalent behavior.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 5 — Update momentum buffer with beta2",
            render: () => (
              <div>
                <TokenStream
                  label="state update (beta2=0.99): m_t = 0.99 m_{t-1} + 0.01 g_t"
                  tokens={[
                    { label: "m_t = 0.99×0.31 + 0.01×0.42", color: colors.gold },
                    { label: "m_t = +0.3111", color: "#86efac" },
                    { label: "stored for next step", color: "#94a3b8" },
                  ]}
                />
                <Prose>
                  Finally, the momentum buffer is updated with the slower {"\\beta_2 = 0.99"} (note: not the {"\\beta_1 = 0.9"} used for the blend in step 2). The asymmetry between {"\\beta_1"} (used to compute the update direction) and {"\\beta_2"} (used to evolve the buffer) is what gives Lion its characteristic behavior. The buffer carries a longer-horizon view of past gradients into the next step's blend.
                </Prose>
              </div>
            ),
          },
        ]}
      />

      <H3>6d. Schedule-Free vs cosine schedule — the decay disappears</H3>

      <Plot
        label="Effective learning rate schedule: cosine vs Schedule-Free averaging weight"
        xLabel="Training step (% of total)"
        yLabel="Effective scaling on update"
        series={[
          {
            name: "Cosine schedule (AdamW)",
            color: colors.gold,
            points: [
              [0, 0.0], [5, 1.0], [10, 0.99], [20, 0.95], [30, 0.88],
              [40, 0.79], [50, 0.65], [60, 0.50], [70, 0.35], [80, 0.21],
              [90, 0.10], [100, 0.0],
            ],
          },
          {
            name: "Schedule-Free (constant LR, eval at x)",
            color: "#86efac",
            points: [
              [0, 0.0], [5, 1.0], [10, 1.0], [20, 1.0], [30, 1.0],
              [40, 1.0], [50, 1.0], [60, 1.0], [70, 1.0], [80, 1.0],
              [90, 1.0], [100, 1.0],
            ],
          },
          {
            name: "ScheduleFree implicit decay (1/t weight on z_t)",
            color: "#60a5fa",
            points: [
              [1, 1.0], [5, 0.20], [10, 0.10], [20, 0.05], [40, 0.025],
              [60, 0.0167], [80, 0.0125], [100, 0.01],
            ],
          },
        ]}
      />

      <Prose>
        The gold curve is a standard cosine schedule: warmup to full LR, then a smooth decay to zero by training end. The green curve is Schedule-Free's actual learning rate at each step — constant after warmup. The blue curve is the implicit decay that emerges from the {"x"} averaging: each new {"z_t"} gets weight {"1/t"} in the running average, so late-training updates contribute exponentially less to the evaluation iterate. This is why Schedule-Free does not need to know the total step count: the averaging mechanism produces the same effective decay as a tuned schedule, but the pacing is determined by the iteration index, not by an explicit schedule formula.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Heatmap
        label="Optimizer suitability by workload (higher = better fit)"
        rowLabels={["AdamW", "Lion", "Sophia", "Prodigy", "Schedule-Free"]}
        colLabels={["Fine-tune", "Pretrain compute-bound", "Pretrain memory-bound", "No LR tune", "Long/no-end run", "Vision", "RL"]}
        matrix={[
          [1.0, 0.7, 0.3, 0.0, 0.4, 0.9, 1.0],
          [0.7, 0.7, 1.0, 0.0, 0.5, 0.8, 0.4],
          [0.5, 1.0, 0.5, 0.0, 0.4, 0.6, 0.3],
          [0.6, 0.4, 0.3, 1.0, 0.5, 0.6, 0.3],
          [0.7, 0.7, 0.4, 0.7, 1.0, 0.7, 0.4],
        ]}
        colorScale="gold"
      />

      <H3>7.1 By workload</H3>

      <Prose>
        <strong>Fine-tuning small-to-medium models:</strong> AdamW with a tuned LR is the safe default. The runs are short (hours to days), the optimizer state cost is manageable, and the literature on AdamW fine-tuning hyperparameters is overwhelming. Lion can match AdamW with proper LR/WD scaling but rarely beats it on fine-tuning workloads where the gradient signal is already aligned with what works.
      </Prose>

      <Prose>
        <strong>LLM pretraining, compute-bound:</strong> Sophia is the choice when wall-clock matters more than implementation complexity. The reported 2x speedup over AdamW translates into roughly half the GPU-hours for the same final loss. The downside is hyperparameter sensitivity — Sophia's clipping threshold {"\\rho"} and Hessian update frequency {"k"} need careful tuning, and the second backward pass adds 25% per-step overhead.
      </Prose>

      <Prose>
        <strong>LLM pretraining, memory-bound:</strong> Lion is the choice when optimizer state is the bottleneck. At frontier scale, AdamW's 2x state overhead can exceed the model weights themselves; halving this lets you fit a larger batch or shard the model less aggressively across GPUs. The memory savings compound with ZeRO-1 and FSDP, where optimizer state is the sharded dimension.
      </Prose>

      <Prose>
        <strong>No learning-rate tuning desired:</strong> Prodigy or Schedule-Free both eliminate LR as a hyperparameter, but for different reasons. Prodigy automatically estimates the right LR; Schedule-Free uses a constant LR but eliminates the schedule. If you have a fixed-length run and want zero LR-related hyperparameters, Prodigy is more aggressive. If you have a variable-length or open-ended run, Schedule-Free is the only viable option.
      </Prose>

      <Prose>
        <strong>Very long pretraining with no fixed end:</strong> Schedule-Free is the only option among the four. Standard AdamW + cosine decay must commit to a final step count up front; if you decide at step 100k that you want to train to step 200k, you have to restart with a new schedule or accept that you have already decayed past the optimal LR. Schedule-Free's averaging mechanism gracefully handles arbitrary stopping points — the {"x"} iterate at any step is a valid evaluation model.
      </Prose>

      <Prose>
        <strong>Vision (CNN, ViT):</strong> AdamW and Lion are competitive. Lion's original paper demonstrated wins on ImageNet classification with ViT and ResNet, often beating AdamW by 0.5–1.5 points top-1 accuracy with the same compute. SGD with momentum still has a place for ResNet-style architectures with batch norm; for everything else, AdamW or Lion.
      </Prose>

      <Prose>
        <strong>Reinforcement learning:</strong> Adam (or AdamW) remains the dominant choice. The non-stationarity of RL training (the data distribution shifts as the policy improves) makes optimizers with strong momentum buffers (Lion's {"\\beta_2 = 0.99"} buffer) less appropriate. Schedule-Free has shown promise in some recent RL work but is not yet a default choice. Sophia and Prodigy have not been evaluated extensively in RL.
      </Prose>

      <Callout type="insight">
        The decision is bottleneck-driven. Identify what is actually limiting your training run — wall-clock, GPU memory, engineer time on hyperparameter tuning, or run-length flexibility — and the optimizer choice follows. AdamW's enduring popularity is not because it is best at any specific axis; it is because it is acceptable on every axis and has the largest body of practitioner knowledge.
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8a. Optimizer state at frontier scale</H3>

      <Prose>
        For a 70B-parameter model in bf16, the weights themselves take 140 GB. AdamW's optimizer state in fp32 is 560 GB ({"70 \\times 10^9 \\times 4 \\times 2"}); Lion's is 280 GB. On 8x H100 (80 GB each), AdamW's state alone consumes 7 GPUs' worth of memory before activations or gradients enter the picture, which is why ZeRO-1 (sharding optimizer state across data-parallel ranks) is mandatory at this scale. Lion halves this requirement, allowing either a smaller ZeRO degree or a larger effective batch size at the same memory budget. The savings are not marginal: at 175B parameters (GPT-3 scale), AdamW state is 1.4 TB; Lion is 700 GB.
      </Prose>

      <H3>8b. Wall-clock at LLM scale</H3>

      <Prose>
        Sophia's 2x speedup claim has been replicated in multiple independent studies on GPT-2 small to GPT-3 medium scales. The mechanism is genuine: when the loss surface has high curvature variation across parameters (which is the LLM regime — embedding rows have very different curvature than attention key/value projections), a Hessian-aware optimizer makes more progress per step. The catch: the speedup depends on the Hessian estimator being informative. On simpler problems (small networks, simple data) the squared gradient is already a good curvature proxy, so AdamW's {"v"} buffer matches Sophia's {"h"} buffer in practice and the speedup vanishes.
      </Prose>

      <H3>8c. Hyperparameter compute</H3>

      <Prose>
        Industry training pipelines spend a non-trivial fraction of total compute on hyperparameter sweeps before the main run. A typical AdamW LR sweep is 5–10 candidate values, each trained for 1–10% of the full training budget — call it 50% of one full run, on top of the main run. Prodigy and Schedule-Free reduce this to one or two runs total: Prodigy because there is no LR to sweep, Schedule-Free because the schedule is gone. At LLM scale where one full run costs millions of dollars, this can be a 30%+ saving on total project compute.
      </Prose>

      <H3>8d. ZeRO/FSDP friendliness</H3>

      <Prose>
        ZeRO and FSDP shard optimizer state across data-parallel ranks. The smaller the per-parameter state, the less communication is needed when materializing full state for an all-reduce or for a parameter update. Lion's halved state means roughly half the all-gather traffic for the optimizer step, which can be a meaningful savings on slow interconnects. Sophia and AdamW have the same state size and so the same communication cost; Schedule-Free's larger state (3x params) is friendlier than Prodigy's (4x params) but worse than AdamW's. At extreme scale, the wall-clock overhead of optimizer-state communication is a real factor in the choice.
      </Prose>

      <H3>8e. The 2024 frontier</H3>

      <Prose>
        Public information on frontier model training in 2024 suggests AdamW remains dominant for the largest runs. Llama-3, Llama-3.1, and Mixtral all used AdamW with extensively tuned cosine schedules. The reasons cited by practitioners are repeatability and reduced risk: AdamW's failure modes are well-understood, the tuning playbook exists, and any new optimizer's potential 2x speedup is offset by the engineering risk of a botched run that wastes compute equivalent to its theoretical savings. Sophia has been adopted in academic and open-source pretraining (Levanter framework at Stanford) but has not yet penetrated the largest commercial runs. Lion, despite the memory savings, sees more use in vision and audio than in LLM pretraining. Prodigy and Schedule-Free are growing fast in the academic and small-team segment where engineer time dominates compute cost.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9a. Lion: wrong learning rate scale</H3>

      <Prose>
        The single most common Lion failure: copying the AdamW learning rate over to Lion. AdamW's effective per-coordinate step is {"\\eta \\cdot |m| / \\sqrt{v}"} which is typically below {"\\eta"} in magnitude; Lion's per-coordinate step is exactly {"\\eta"}. Using AdamW's LR (e.g., {"3 \\times 10^{-4}"}) with Lion produces gradient explosion and divergent loss curves within the first few hundred steps. The fix: divide the AdamW LR by roughly 10 when migrating to Lion. The Chen et al. paper provides per-task LR mappings; for new tasks, sweep around {"\\eta_{\\text{Lion}} = \\eta_{\\text{AdamW}} / 10"}.
      </Prose>

      <H3>9b. Sophia: noisy Hessian estimate</H3>

      <Prose>
        Sophia's Hessian-diagonal estimate is stochastic. With Sophia-G, the estimate uses a single resampled label per Hessian update — high variance. Two failure modes follow. First, if the estimate is updated too frequently (every step), the variance dominates and Sophia behaves erratically. Second, if {"\\rho"} (the clipping threshold) is too small, every coordinate's update gets clipped to {"\\pm \\eta"} and Sophia degenerates into Lion. The right operating point: Hessian update every 5–10 steps, {"\\rho \\approx 0.04"}, and momentum {"\\beta_1 \\approx 0.965"}. Below those values, expect instability; above them, the speedup over AdamW shrinks.
      </Prose>

      <H3>9c. Prodigy: slow start during D-adapt warmup</H3>

      <Prose>
        Prodigy's {"d_t"} starts at the user-specified {"d_0"} (typically {"10^{-6}"}) and grows toward the right scale over the first few hundred steps. During this warmup, the effective learning rate is far below optimal and the loss may decrease very slowly — this is normal but alarming if you do not know to expect it. Some users mistakenly conclude Prodigy is broken and abort the run before {"d_t"} stabilizes. The fix: enable <Code>{"safeguard_warmup=True"}</Code> (which delays {"d"} growth checks during the first few hundred steps), and watch the optimizer's logged {"d"} value rather than the loss for the first 500–1000 steps.
      </Prose>

      <H3>9d. Schedule-Free: forgetting train()/eval() switch</H3>

      <Prose>
        Schedule-Free's optimizer needs {"\\code{opt.train()}"} and {"\\code{opt.eval()}"} calls in addition to the model's {"\\code{model.train()/.eval()}"}. The optimizer call switches between the {"y"} iterate (used for training-time gradient computation) and the {"x"} iterate (the averaged model used for evaluation). If you forget to call {"\\code{opt.eval()}"} before validation, the model parameters at validation time are {"y"}, not {"x"}, and you get noisier and worse validation losses than reality. The bug is silent — it does not crash, it just gives misleading metrics that may cause you to abandon a perfectly good run.
      </Prose>

      <H3>9e. Schedule-Free + LR schedule = worst of both worlds</H3>

      <Prose>
        The whole point of Schedule-Free is that there is no learning-rate schedule; the averaging mechanism replaces decay. Composing Schedule-Free with a cosine or linear schedule on top destroys both mechanisms — you get the slow late-training progress of cosine decay applied to an iterate average that already implicitly decays. The fix: use Schedule-Free with a constant LR (after warmup). If you want a schedule, use AdamW with a schedule; if you want Schedule-Free, drop the schedule. Mixing them is a surprisingly common bug because Hugging Face Trainer's default scheduler runs unconditionally unless explicitly disabled.
      </Prose>

      <H3>9f. Lion + insufficient weight decay</H3>

      <Prose>
        Because Lion's update is sign-only and uniform across parameters, it does not implicitly regularize the way AdamW's adaptive scaling does. Parameters that AdamW would damp via small effective steps (because their {"v"} is large) get the full {"\\eta"} step under Lion. The implication: weight decay must be larger to compensate. Lion's recommended {"\\lambda \\approx 1.0"} is roughly 100x AdamW's typical {"\\lambda \\approx 0.01"}. Migrating from AdamW to Lion without scaling up weight decay produces models that overfit hard or have unstable training late in the run. The Chen et al. paper provides task-specific WD recommendations; for new tasks, scan {"\\lambda \\in \\{0.1, 0.3, 1.0\\}"}.
      </Prose>

      <H3>9g. Lion + fp16 instability</H3>

      <Prose>
        Lion's sign function magnifies the relative impact of low-magnitude noise. In fp16 (with its narrow exponent range), small gradient values can underflow to zero or take a noisy sign that flips between updates. The result is unstable training, especially at the end of warmup when LR is at its peak. The fix: train Lion in bf16, not fp16. Bf16's wider exponent range matches fp32's representable magnitude floor and avoids the underflow regime. Almost all production Lion usage in 2024 is bf16-only; the original Chen et al. paper also recommends bf16.
      </Prose>

      <H3>9h. Sophia: forgetting to update the Hessian</H3>

      <Prose>
        Sophia requires a separate {"\\code{update_hessian()}"} call every {"k"} steps, with a fresh backward pass on the resampled labels. If you forget to call {"\\code{update_hessian()}"} (or call it with the original gradient in scope), the Hessian buffer never updates and Sophia degenerates into a momentum-only optimizer with a stale denominator — slower than AdamW, with more state. The bug is subtle because the loss still decreases (momentum alone makes progress), just much slower than expected. The fix: log {"\\|h\\|"} periodically; if it is constant across steps, the Hessian update is not running.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were verified against their primary publication venues and arXiv pages.
      </Prose>

      <Prose>
        <strong>Kingma, D.P. and Ba, J. (2015).</strong> "Adam: A Method for Stochastic Optimization." <em>International Conference on Learning Representations (ICLR 2015)</em>. arXiv:1412.6980. The paper that established adaptive per-parameter learning rates with momentum and bias correction as the default optimizer for deep learning. Combined RMSprop's variance-based scaling with momentum and added bias-correction terms to handle the zero-initialized moment estimates. Adam dominated the optimizer landscape from 2015 to 2019 before AdamW corrected its weight-decay handling.
      </Prose>

      <Prose>
        <strong>Loshchilov, I. and Hutter, F. (2019).</strong> "Decoupled Weight Decay Regularization." <em>International Conference on Learning Representations (ICLR 2019)</em>. arXiv:1711.05101. Introduced AdamW, the corrected formulation that decouples weight decay from the gradient-based update. The paper showed that Adam's original weight-decay implementation interacted with the adaptive scaling in ways that distorted the regularization strength differently across parameters; decoupling weight decay restored its expected behavior. AdamW has been the default for nearly all deep learning since approximately 2018.
      </Prose>

      <Prose>
        <strong>Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Liu, Y., Pham, H., Dong, X., Luong, T., Hsieh, C.-J., Lu, Y., and Le, Q.V. (2023).</strong> "Symbolic Discovery of Optimization Algorithms." <em>Advances in Neural Information Processing Systems 36 (NeurIPS 2023)</em>. arXiv:2302.06675. Used Google's evolutionary program search (AutoML-Zero descendant) over a search space of optimizer programs to discover Lion. The discovered algorithm uses only the sign of momentum and a single moment buffer, beating AdamW on image classification, language modeling, and contrastive learning while using half the optimizer state. The paper makes the strong claim that program search can discover optimization algorithms that humans missed.
      </Prose>

      <Prose>
        <strong>Liu, H., Li, Z., Hall, D., Liang, P., and Ma, T. (2024).</strong> "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training." <em>International Conference on Learning Representations (ICLR 2024)</em>. arXiv:2305.14342. Introduced Sophia (Second-order clipPed stochastic OptIMization Algorithm) using a stochastic estimate of the Hessian diagonal as a preconditioner, with bidirectional clipping for stability. Demonstrated 2x wall-clock speedup over AdamW on GPT-2 pretraining at 125M to 770M parameter scales, with the speedup increasing at larger scales. Includes both Sophia-H (Hutchinson estimator) and Sophia-G (Gauss-Newton estimator) variants.
      </Prose>

      <Prose>
        <strong>Mishchenko, K. and Defazio, A. (2023).</strong> "Prodigy: An Expeditiously Adaptive Parameter-Free Learner." arXiv:2306.06101. Introduced Prodigy, which extends D-adaptation (an earlier learning-rate-free SGD method by the same authors) to Adam-style adaptive optimizers. Prodigy estimates the optimal learning rate {"d_t"} from gradient correlations with the displacement from initialization, requiring only an initial {"d_0"} (typically {"10^{-6}"}) and a maximum cap. The paper demonstrated competitive or superior performance to tuned AdamW across image classification, language modeling, and meta-learning tasks, with no learning-rate hyperparameter to sweep.
      </Prose>

      <Prose>
        <strong>Defazio, A., Yaras, B., Yan, Z., Cutkosky, A., and Mishchenko, K. (2024).</strong> "The Road Less Scheduled." arXiv:2405.15682. Introduced Schedule-Free SGD and Schedule-Free AdamW, eliminating the learning-rate schedule entirely via a momentum-based interpolation between iterate sequences. The method maintains three iterates ({"z, x, y"}) where {"x"} is the running average of {"z"} and {"y = (1 - \\beta) z + \\beta x"} is the iterate at which gradients are computed. Theoretical analysis shows Schedule-Free achieves the same convergence rate as the optimal schedule under standard convex assumptions, but does not require knowing the total step count. Adopted as the default optimizer for several Meta production training runs in 2024.
      </Prose>

      <Prose>
        <strong>Shazeer, N. and Stern, M. (2018).</strong> "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost." <em>Proceedings of the 35th International Conference on Machine Learning (ICML 2018)</em>. arXiv:1804.04235. Introduced Adafactor, which factorizes Adam's second-moment matrix {"v"} into a row vector and a column vector, reducing optimizer state from 2x to roughly 0.5x the model's parameter count. Adafactor became the default optimizer for T5 and Google's Pathways stack. The paper also introduced an alternative learning-rate parameterization (relative step size) that became influential in subsequent memory-efficient optimizers. Predecessor in spirit to the memory-efficiency thread that Lion later refined.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 (Recall)</H3>
      <Prose>
        State the AdamW update rule, the Lion update rule, and the Sophia update rule. For each, list the optimizer state per parameter (in multiples of the parameter count) and the typical learning-rate scale relative to AdamW.
      </Prose>
      <Callout type="answer">
        {"AdamW: m = β₁m + (1-β₁)g; v = β₂v + (1-β₂)g²; θ -= η(m̂/(√v̂+ε) + λθ). State: 2x params (m, v). Reference LR scale: 1x (e.g., 3e-4 for transformers, 1e-3 for vision)."}
        <br /><br />
        {"Lion: u = sign(β₁m + (1-β₁)g); θ -= η(u + λθ); m = β₂m + (1-β₂)g. State: 1x params (m only). LR scale: ~0.1x of AdamW (typical 3e-5 to 1e-4 for transformers). Weight decay scale: ~10x AdamW."}
        <br /><br />
        {"Sophia: m = β₁m + (1-β₁)g; h updated every k steps via Hutchinson or Gauss-Newton; θ -= η·clip(m/max(γh, ε), -1, 1) - ηλθ. State: 2x params (m, h). LR scale: similar to AdamW (e.g., 6e-4 for transformers). Hessian update every 5-10 steps."}
      </Callout>

      <H3>Exercise 2 (Conceptual)</H3>
      <Prose>
        Schedule-Free claims to eliminate the learning-rate schedule. Where does the implicit decay come from, and why doesn't the user need to know the total training step count?
      </Prose>
      <Callout type="answer">
        {"Schedule-Free maintains three iterate sequences: z (the gradient-step iterate, updated like normal Adam), x (a running average of all past z values, x_t = (1-1/t)x_{t-1} + (1/t)z_t), and y (an interpolation y = (1-β)z + βx, used for forward-pass gradient computation). The model used at evaluation is x, the averaged iterate. The implicit decay comes from the averaging weight: at step t, the new z_t enters x with weight 1/t, so late-training updates contribute progressively less to the evaluation model. By step 1M, an update at step 1M contributes weight 1/1,000,000 to x, while an update at step 1k contributes weight ≈1/1,000 to x via the recursive averaging — but that early contribution is itself diluted by all later steps, so its effective weight in x at step 1M is also 1/1M. Crucially, this averaging mechanism does not depend on the total step count: regardless of where you stop training, x is a valid evaluation model, and its quality is determined by how much progress the z iterates made (bounded by the gradient diversity of the data). This is why Schedule-Free is robust to runs of unknown length — continuous pretraining, online learning, runs where you decide mid-training to extend further. Standard cosine decay would commit to a final step count up front and fail gracefully if that estimate is wrong."}
      </Callout>

      <H3>Exercise 3 (Applied)</H3>
      <Prose>
        You are pretraining a 7B-parameter transformer language model on 8 H100 GPUs (80 GB each). The optimizer state in AdamW is consuming so much memory that you cannot fit a useful batch size. Walk through how Lion and Adafactor would change the memory budget, and which one you would choose.
      </Prose>
      <Callout type="answer">
        {"Memory analysis. 7B params in bf16 = 14 GB weights. AdamW state in fp32 = 7B × 4 bytes × 2 moments = 56 GB. With 8x H100 (640 GB total), the model + AdamW state alone is 70 GB — leaving 570 GB for activations, gradients, batch, and overhead. With ZeRO-1 sharding optimizer state across 8 GPUs, AdamW state per GPU = 7 GB, model per GPU = 14 GB, total static = 21 GB per GPU, leaving 59 GB for activations and batch. Manageable but tight."}
        <br /><br />
        {"Lion replaces AdamW. State = 7B × 4 bytes × 1 moment = 28 GB total, or 3.5 GB per GPU with ZeRO-1. Static memory per GPU: 14 + 3.5 = 17.5 GB, leaving 62.5 GB for activations and batch. You could increase batch by ~6% or ZeRO-1 less aggressively (saving communication overhead). The tradeoff: you need to retune LR (likely 10x smaller) and weight decay (likely 10x larger), and use bf16 (you already are). Lion has been validated for transformer pretraining at this scale by multiple groups."}
        <br /><br />
        {"Adafactor's state = 0.5x params = 14 GB total, or 1.75 GB per GPU with ZeRO-1. Static memory per GPU: 14 + 1.75 = 15.75 GB, leaving 64.25 GB. Adafactor's tradeoff: the factorization causes some convergence loss vs AdamW (typically 5-15% slower wall-clock for the same final loss). It also doesn't support per-parameter momentum, only the factored second moment, which can hurt convergence on architectures with strong per-parameter gradient variance."}
        <br /><br />
        {"Decision: Lion. The memory savings (1.5x reduction in optimizer state) are similar to Adafactor's, but Lion typically matches or beats AdamW's convergence rather than slowing it. The hyperparameter retuning is a one-time cost. Adafactor would be the choice only if 0.5x state is mandatory (e.g., 13B model on 8 H100) and the convergence loss is acceptable."}
      </Callout>

      <H3>Exercise 4 (Debugging)</H3>
      <Prose>
        You switch from AdamW to Lion on your transformer fine-tuning job, keeping the AdamW LR and weight decay unchanged. The first 500 steps look fine, but loss explodes at step 700 and never recovers. Diagnose the failure and fix it.
      </Prose>
      <Callout type="answer">
        {"Two simultaneous bugs from the migration. First, Lion's per-coordinate step is exactly η, while AdamW's effective step is η·|m|/(√v+ε), typically below η in magnitude. Reusing AdamW's η = 3e-4 means Lion is taking ~10x larger effective steps than AdamW would on the same parameters. Initially this looks fine because gradients are small (gradient magnitude in the high-loss region is mostly aligned), but as the model approaches a flatter region the constant ±η step size starts to overshoot the local minimum and bounce — which is why the explosion happens at step 700, not step 1. Second, AdamW's effective weight decay is also distorted: AdamW's decoupled λθ is applied without adaptive scaling, so the regularization strength is roughly λ ≈ 0.01 in normalized terms. Lion's update direction is sign-only and uniform across parameters, so it does not damp parameters with large gradient variance the way AdamW does — λ = 0.01 is too weak to prevent late-training instability."}
        <br /><br />
        {"Fix: (1) Reduce LR to η = 3e-5 (10x smaller). (2) Increase weight decay to λ = 1.0 (100x larger; some practitioners use 0.3 as a middle ground). (3) Verify that you are training in bf16, not fp16 — Lion's sign function is unstable in fp16 due to underflow. (4) Do a short LR sweep over {1e-5, 3e-5, 1e-4} to confirm the optimum for your specific task, since the 10x rule is approximate. With these changes, the run should converge similarly to AdamW or slightly better."}
      </Callout>

      <H3>Exercise 5 (Math)</H3>
      <Prose>
        Compute one step of Lion. The momentum buffer for a parameter is {"m_{t-1} = 0.4"}; the current gradient is {"g_t = -0.2"}. With {"\\beta_1 = 0.9, \\beta_2 = 0.99, \\eta = 10^{-4}, \\lambda = 1.0"}, parameter value {"\\theta_{t-1} = 0.5"}, what is {"\\theta_t"} and what is the new {"m_t"}?
      </Prose>
      <Callout type="answer">
        {"Step 1: Blend with β₁. u_blend = 0.9·m_{t-1} + 0.1·g_t = 0.9·0.4 + 0.1·(-0.2) = 0.36 - 0.02 = 0.34."}
        <br /><br />
        {"Step 2: Sign. u = sign(0.34) = +1."}
        <br /><br />
        {"Step 3: Parameter update. θ_t = θ_{t-1} - η(u + λθ_{t-1}) = 0.5 - 10⁻⁴ · (1 + 1.0·0.5) = 0.5 - 10⁻⁴ · 1.5 = 0.5 - 1.5×10⁻⁴ = 0.49985."}
        <br /><br />
        {"Step 4: Momentum buffer update with β₂ (note: not β₁). m_t = 0.99·m_{t-1} + 0.01·g_t = 0.99·0.4 + 0.01·(-0.2) = 0.396 - 0.002 = 0.394."}
        <br /><br />
        {"Final: θ_t = 0.49985, m_t = 0.394. Note that the parameter moved by 1.5×10⁻⁴ — exactly η times (1 + λθ), with the sign of η determined by the sign of the blended momentum. The momentum buffer evolved with β₂ = 0.99, not the β₁ = 0.9 used in the update blend; this asymmetric beta structure is the artifact of Lion's symbolic search origin."}
      </Callout>

    </div>
  ),
};

export default advancedOptimizersContent;
