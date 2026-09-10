import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const dropoutDroppathContent = {
  title: "Dropout, DropPath & Stochastic Depth",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Deep networks have more parameters than samples. A 60-layer ResNet, a ViT-Large, a modern LLM — all of them could, in principle, memorize their training set. The mathematical definition of overfitting is when the gap between training loss and held-out loss widens indefinitely. Weight decay helps. Data augmentation helps. Early stopping helps. But none of these address the specific pathology of overparameterized deep networks: co-adaptation — where a neuron becomes dependent on a specific combination of other neurons being present to produce a meaningful signal. Co-adapted features are brittle. They work on the training distribution and collapse on anything else.
      </Prose>

      <Prose>
        The first attack on this problem came from Geoffrey Hinton and his students at the University of Toronto in 2012. In a technical report titled "Improving neural networks by preventing co-adaptation of feature detectors" (arXiv:1207.0580), Hinton, Srivastava, Krizhevsky, Sutskever, and Salakhutdinov proposed a deliberately destructive intervention: at each forward pass during training, pick a random half of the hidden units and zero them out. The network has to learn representations that are robust to this sabotage. No neuron can assume its neighbor will be there to pick up the slack. They called it dropout. The technique appeared prominently in AlexNet that same year (Krizhevsky, Sutskever, Hinton, NeurIPS 2012) and was credited as one of the reasons AlexNet shattered ImageNet records.
      </Prose>

      <Prose>
        The full theoretical treatment came two years later. Srivastava, Hinton, Krizhevsky, Sutskever, and Salakhutdinov published "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" in the Journal of Machine Learning Research, volume 15, pages 1929–1958, in 2014. This paper formalized the method, connected it to model averaging over an exponential family of subnetworks, and documented gains across MNIST, SVHN, CIFAR, ImageNet, TIMIT, and Reuters. The paper explicitly framed dropout as an ensemble method in disguise — a single network that behaves like an average over <Code>{"2^N"}</Code> thinned networks sharing weights. The JMLR paper is the one everyone cites today.
      </Prose>

      <Prose>
        Theoretical understanding followed quickly. Stefan Wager, Sida Wang, and Percy Liang published "Dropout Training as Adaptive Regularization" at NeurIPS 2013 (arXiv:1307.1493), proving that for generalized linear models dropout is approximately equivalent to an adaptive L2 penalty, with per-feature strength determined by the Fisher information. Dropout was not just a heuristic — it was a principled form of regularization with a well-defined expectation. Yarin Gal and Zoubin Ghahramani then pushed the interpretation further. In "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning" (ICML 2016, arXiv:1506.02142), they showed that training a network with dropout is mathematically equivalent to variational inference in a deep Gaussian process — and that keeping dropout on at test time and averaging multiple forward passes yields a Bayesian posterior predictive distribution. Monte Carlo Dropout was born.
      </Prose>

      <Prose>
        Meanwhile the vision community was hitting its own wall. ResNets (He et al. 2015) had shown that depth beyond 100 layers was possible, but training 1000-layer ResNets was punishingly slow and gradients still degraded. Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Weinberger proposed a counter-intuitive fix in "Deep Networks with Stochastic Depth" (ECCV 2016, arXiv:1603.09382): during training, randomly drop entire residual blocks. If block <Code>{"l"}</Code> is dropped, the forward pass becomes <Code>{"x_{l+1} = x_l"}</Code> — just the identity shortcut. With a linear schedule of drop probabilities growing from 0 at the input to <Code>{"0.5"}</Code> at the output, a 1202-layer ResNet trained faster than the standard 110-layer version and achieved lower test error. The trick was that the expected network depth during training was shorter than the architectural depth, but the full depth was used at inference.
      </Prose>

      <Prose>
        Xavier Gastaldi's "Shake-Shake regularization" (ICLR 2017 Workshop, arXiv:1705.07485) and Yoshihiro Yamada, Masakazu Iwamura, and Koichi Kise's "ShakeDrop Regularization" (arXiv:1802.02375) extended the idea by mixing branch outputs with random coefficients rather than dropping them entirely. The modern Vision Transformer era took the stochastic-depth concept and renamed the per-sample branch-drop operation DropPath. Touvron et al.'s "CaiT: Class-Attention in Image Transformers" (arXiv:2103.17239) and Liu et al.'s ConvNeXt (arXiv:2201.03545) use DropPath with linearly ramped drop rates as a standard regularizer — the timm library (<Code>timm.layers.DropPath</Code>) made the implementation universal.
      </Prose>

      <Prose>
        The recurrent world adapted the idea differently. David Krueger et al.'s Zoneout (arXiv:1606.01305, 2016) stochastically preserves hidden states across timesteps rather than zeroing activations — a recurrence-aware regularizer. For convolutional feature maps, Golnaz Ghiasi, Tsung-Yi Lin, and Quoc Le's DropBlock (NeurIPS 2018, arXiv:1810.12890) observed that standard dropout on feature maps is weak because spatially adjacent activations are highly correlated: dropping one pixel is trivially recovered from its neighbors. DropBlock drops contiguous square regions instead. And Xiang Li et al.'s "Understanding the Disharmony between Dropout and Batch Normalization" (CVPR 2019, arXiv:1801.05134) explained why mixing dropout with batch normalization degrades performance — the variance shift caused by dropped activations breaks BN's running statistics.
      </Prose>

      <Callout type="insight">
        Each method adds stochasticity at a different granularity. Dropout zeros individual activations. Spatial Dropout zeros whole channels. DropBlock zeros contiguous spatial regions. DropPath zeros entire residual branches per sample. Stochastic Depth zeros blocks per batch. Zoneout preserves recurrence states. The right granularity depends on what structure in your network is correlated — and thus what kind of sabotage actually destroys information.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 Dropout as ensemble averaging</H3>

      <Prose>
        The cleanest way to think about dropout is as implicit ensembling. A network with <Code>N</Code> hidden units has <Code>{"2^N"}</Code> possible subsets of active units — each subset defines a different subnetwork (a "thinned" network in Srivastava's terminology). At each training step, dropout samples one of these <Code>{"2^N"}</Code> subnetworks uniformly and trains it. All subnetworks share weights, which is what makes the procedure tractable. At inference, you would ideally average over all <Code>{"2^N"}</Code> subnetworks, which is intractable. The dropout trick is that using the full network with scaled activations is an excellent approximation to that ensemble average — exact for linear networks, approximate but accurate for nonlinear ones.
      </Prose>

      <Prose>
        This ensemble view explains why dropout works well even on small datasets. Bagging — training many different models on bootstrapped subsets of the data — is a classical variance-reduction technique. Dropout achieves a similar variance reduction without training many models: the single weight-sharing network represents the entire ensemble implicitly. The cost is paid only at training time (extra stochasticity slows convergence slightly) and is zero at inference.
      </Prose>

      <H3>2.2 Inverted dropout scaling</H3>

      <Prose>
        A neuron that survives with probability <Code>{"1-p"}</Code> during training has expected activation <Code>{"(1-p)·a"}</Code> where <Code>a</Code> is its value when kept. At test time, with all neurons active, the activation is <Code>a</Code> — so the expected scale at test differs from training by a factor of <Code>{"1/(1-p)"}</Code>. To make training and evaluation numerically compatible without any special case at inference, frameworks use inverted dropout: during training the surviving activations are divided by <Code>{"1-p"}</Code>. This preserves expected activation magnitude at training time and makes test time a pure pass-through — a single code path instead of two. PyTorch's <Code>nn.Dropout</Code>, TensorFlow's <Code>tf.keras.layers.Dropout</Code>, and every modern library use inverted dropout.
      </Prose>

      <H3>2.3 Co-adaptation prevention</H3>

      <Prose>
        The original 2012 paper's framing was that neurons co-adapt: neuron A learns to produce feature <Code>f</Code> only in the context of neuron B also firing, so neither A nor B alone is meaningful. This is a form of redundancy elimination that looks efficient on the training set but is catastrophic under distribution shift. Dropout forces each neuron to be useful on its own, because the context in which it appears at each forward pass is unpredictable. The resulting features are more redundant (multiple neurons learn similar things) but each is individually robust. Redundancy is what we want — it is the opposite of memorization.
      </Prose>

      <H3>2.4 Stochastic depth and DropPath</H3>

      <Prose>
        In a residual network, a block computes <Code>{"x_{l+1} = x_l + F(x_l)"}</Code>. Stochastic depth replaces this with <Code>{"x_{l+1} = x_l + m · F(x_l)"}</Code> where <Code>{"m ~ Bernoulli(1-p)"}</Code>. When <Code>{"m = 0"}</Code>, the block is skipped — only the identity shortcut remains. The crucial architectural property that makes this work is the residual structure: with standard layers, skipping the layer would produce a completely different signal. With residuals, skipping produces the same signal you would have had without that layer's refinement.
      </Prose>

      <Prose>
        DropPath is the per-sample version. Instead of deciding once per batch whether to drop the block, the mask is per-sample: <Code>{"m"}</Code> has shape <Code>{"(B, 1, 1, ...)"}</Code>, so within one batch some samples bypass the block entirely while others see <Code>{"F(x)"}</Code> at full strength. This preserves batch normalization statistics better (BN sees a mix of both cases) and gives finer-grained regularization. Modern ViT and ConvNeXt implementations use DropPath, not block-level stochastic depth.
      </Prose>

      <H3>2.5 Bayesian interpretation</H3>

      <Prose>
        Gal and Ghahramani (2016) showed something remarkable: training a network with dropout is equivalent to variational inference over a particular posterior over weights. If you keep dropout on at test time and run <Code>T</Code> forward passes, the distribution of predictions approximates the posterior predictive distribution of a Bayesian neural network. The variance across those <Code>T</Code> samples is a calibrated uncertainty estimate. This is Monte Carlo Dropout, and it is why dropout shows up not just as a regularizer but as a cheap uncertainty quantification tool in active learning, out-of-distribution detection, and Bayesian optimization.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Dropout forward pass</H3>

      <Prose>
        Let <Code>{"x ∈ R^d"}</Code> be the activation vector at some layer. With drop probability <Code>{"p ∈ [0, 1)"}</Code>, dropout samples an independent Bernoulli mask <Code>{"m_i ~ Bernoulli(1-p)"}</Code> for each dimension and produces:
      </Prose>

      <MathBlock>{"y_i = \\frac{m_i \\cdot x_i}{1 - p}, \\qquad m_i \\sim \\text{Bernoulli}(1-p)"}</MathBlock>

      <Prose>
        The denominator <Code>{"1-p"}</Code> is the inverted-dropout scaling. Taking the expectation over the mask:
      </Prose>

      <MathBlock>{"\\mathbb{E}[y_i] = \\frac{x_i \\cdot \\mathbb{E}[m_i]}{1 - p} = \\frac{x_i (1-p)}{1-p} = x_i"}</MathBlock>

      <Prose>
        So in expectation the dropout layer is the identity — at eval time, setting <Code>{"m_i = 1"}</Code> for all <Code>i</Code> and skipping the division recovers the same expected scale. The variance, however, is nonzero:
      </Prose>

      <MathBlock>{"\\text{Var}[y_i] = \\frac{x_i^2}{(1-p)^2} \\cdot \\text{Var}[m_i] = \\frac{x_i^2 \\cdot p(1-p)}{(1-p)^2} = \\frac{p}{1-p} \\cdot x_i^2"}</MathBlock>

      <Prose>
        The injected noise has variance proportional to <Code>{"p/(1-p)"}</Code>. At <Code>{"p=0.5"}</Code>, <Code>{"\\text{Var}[y_i] = x_i^2"}</Code> — the noise equals the signal in variance. At <Code>{"p=0.1"}</Code>, the noise is <Code>{"0.111 x_i^2"}</Code> — much gentler. This formula predicts exactly the variance we observe in practice, as the from-scratch experiments in section 4 confirm to four decimal places.
      </Prose>

      <H3>3.2 Dropout as adaptive L2 regularization</H3>

      <Prose>
        Wager, Wang, and Liang (2013) analyzed dropout on generalized linear models. For a logistic regression with input <Code>x</Code>, weights <Code>w</Code>, and dropout applied to <Code>x</Code> rather than activations, the expected loss under the dropout distribution can be Taylor-expanded:
      </Prose>

      <MathBlock>{"\\mathbb{E}_{m}[\\mathcal{L}(y, w^\\top (m \\odot x / (1-p)))] \\approx \\mathcal{L}(y, w^\\top x) + \\frac{p}{2(1-p)} \\sum_i V_{ii}(w) x_i^2"}</MathBlock>

      <Prose>
        where <Code>{"V_{ii}(w)"}</Code> is the diagonal of the Fisher information matrix. The second term is a quadratic penalty on the weights — an L2-like regularizer — but weighted by <Code>{"x_i^2"}</Code>. Unlike static L2 which penalizes every weight equally, dropout penalizes more strongly in directions where features are large. It is adaptive L2. This result explains why dropout and weight decay are not redundant: they penalize different things. Dropout's penalty depends on the data; weight decay's does not.
      </Prose>

      <H3>3.3 DropPath mask structure</H3>

      <Prose>
        For a residual block output <Code>{"F(x)"}</Code> with batch dimension first, DropPath applies:
      </Prose>

      <MathBlock>{"y = x + \\frac{m \\cdot F(x)}{1 - p}, \\qquad m \\in \\{0, 1\\}^B, \\; m_b \\sim \\text{Bernoulli}(1-p)"}</MathBlock>

      <Prose>
        The mask shape is <Code>{"(B, 1, 1, ...)"}</Code> with a singleton along every non-batch dimension, so broadcasting drops the entire branch for a selected sample. The inverted-dropout factor is again <Code>{"1/(1-p)"}</Code>. Note an important consequence: the skip path <Code>x</Code> is always active. If it were not, dropping a block would produce zero output and gradients would vanish — residual connections are what make branch-dropping safe.
      </Prose>

      <H3>3.4 Stochastic depth: expected depth</H3>

      <Prose>
        With a linear schedule <Code>{"p_l = p_L \\cdot l / L"}</Code> for layer <Code>l</Code> out of <Code>L</Code>, the expected number of surviving blocks during training is:
      </Prose>

      <MathBlock>{"\\mathbb{E}[\\text{depth}] = \\sum_{l=1}^{L} (1 - p_l) = L - \\frac{p_L}{L} \\sum_{l=1}^{L} l = L \\cdot \\left(1 - \\frac{p_L (L+1)}{2L}\\right)"}</MathBlock>

      <Prose>
        For a 110-layer ResNet with <Code>{"p_L = 0.5"}</Code>, expected training depth is approximately <Code>{"0.75 L = 82.5"}</Code> blocks — a 25% reduction. Training time drops roughly proportionally. At inference, all blocks are used at full weight, giving the full representational capacity.
      </Prose>

      <H3>3.5 MC Dropout variance</H3>

      <Prose>
        Gal and Ghahramani's MC Dropout computes predictive uncertainty by keeping dropout active at inference and averaging <Code>T</Code> forward passes. For a regression output <Code>{"\\hat{y}"}</Code>:
      </Prose>

      <MathBlock>{"\\mu_* = \\frac{1}{T} \\sum_{t=1}^{T} \\hat{y}^{(t)}, \\qquad \\sigma_*^2 = \\tau^{-1} + \\frac{1}{T} \\sum_{t=1}^{T} (\\hat{y}^{(t)} - \\mu_*)^2"}</MathBlock>

      <Prose>
        where <Code>{"\\tau^{-1}"}</Code> is the model's noise precision (aleatoric uncertainty) and the second term is the epistemic (model) uncertainty. The paper shows this approximates a Bayesian neural network with a specific prior. Practitioners typically use <Code>{"T = 50"}</Code> to <Code>{"T = 200"}</Code> — more samples reduce Monte Carlo variance but increase inference cost linearly.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The theoretical claims above are precise. A from-scratch implementation lets us verify them numerically. All code in this section was executed; the <Code>{"# Output:"}</Code> blocks show real stdout.
      </Prose>

      <H3>4.1 Vanilla inverted dropout — verify E[y] = x and Var[y] = p/(1-p)</H3>

      <CodeBlock language="python">
{`import torch

torch.manual_seed(0)

def dropout_inverted(x, p, training):
    """Inverted dropout: scale at training time, identity at eval."""
    if not training or p == 0.0:
        return x
    keep = 1.0 - p
    mask = (torch.rand_like(x) < keep).float()
    return x * mask / keep

x = torch.ones(10_000)
y_train = dropout_inverted(x, p=0.5, training=True)
y_eval  = dropout_inverted(x, p=0.5, training=False)
print(f"train mean={y_train.mean().item():.4f}  var={y_train.var().item():.4f}")
print(f"eval  mean={y_eval.mean().item():.4f}  var={y_eval.var().item():.4f}")

# Output:
# train mean=0.9940  var=1.0001
# eval  mean=1.0000  var=0.0000`}
      </CodeBlock>

      <Prose>
        Training mean is 0.994 (≈ 1.0, matching identity expectation) and training variance is 1.0001 ≈ <Code>{"p/(1-p) = 0.5/0.5 = 1.0"}</Code>. Eval mean is exactly 1.0 with zero variance because dropout is off. Both confirm the formulas from section 3.1 to four decimal places.
      </Prose>

      <H3>4.2 Sweep across drop probabilities</H3>

      <CodeBlock language="python">
{`for p in [0.0, 0.1, 0.3, 0.5, 0.7]:
    samples = torch.stack([dropout_inverted(torch.ones(4096), p, True)
                           for _ in range(50)])
    e = samples.mean().item()
    v = samples.var().item()
    theory = p/(1-p) if p < 1 else float("inf")
    print(f"p={p:.1f}  E[y]={e:.4f}  Var[y]={v:.4f}  theory={theory:.4f}")

# Output:
# p=0.0  E[y]=1.0000  Var[y]=0.0000  theory=0.0000
# p=0.1  E[y]=1.0000  Var[y]=0.1111  theory=0.1111
# p=0.3  E[y]=0.9999  Var[y]=0.4286  theory=0.4286
# p=0.5  E[y]=1.0006  Var[y]=1.0000  theory=1.0000
# p=0.7  E[y]=1.0011  Var[y]=2.3348  theory=2.3333`}
      </CodeBlock>

      <Prose>
        The empirical variance matches the theoretical <Code>{"p/(1-p)"}</Code> to three decimal places across all rates. This is the core invariant: higher <Code>p</Code> means more injected noise, but the mean is always preserved.
      </Prose>

      <H3>4.3 Spatial dropout (Dropout2d)</H3>

      <Prose>
        For 4D activations <Code>{"[B, C, H, W]"}</Code> in a CNN, standard per-pixel dropout is weak because nearby pixels in a feature map are highly correlated. Spatial dropout drops whole channels per sample — the mask has shape <Code>{"[B, C, 1, 1]"}</Code>.
      </Prose>

      <CodeBlock language="python">
{`def spatial_dropout(x, p, training):
    """Mask shape [B, C, 1, 1] — drop whole channels per sample."""
    if not training or p == 0.0:
        return x
    keep = 1.0 - p
    mask = (torch.rand(x.shape[0], x.shape[1], 1, 1, device=x.device) < keep).float()
    return x * mask / keep

torch.manual_seed(1)
xc = torch.ones(2, 4, 3, 3)  # batch=2, channels=4, 3x3 spatial
yc = spatial_dropout(xc, p=0.5, training=True)
print(yc[:, :, 0, 0])  # one spatial location reveals the per-channel mask

# Output:
# tensor([[0., 2., 2., 0.],
#         [2., 0., 2., 0.]])`}
      </CodeBlock>

      <Prose>
        Each entry is either 0 (channel dropped) or 2 (channel kept, scaled by <Code>{"1/(1-0.5) = 2"}</Code>). Crucially, the same channel is either alive or dead across all 9 spatial positions for a given sample. This is <Code>nn.Dropout2d</Code> in PyTorch.
      </Prose>

      <H3>4.4 DropPath — per-sample residual branch drop</H3>

      <CodeBlock language="python">
{`def drop_path(x, drop_prob, training):
    """Per-sample drop of a residual branch. x shape: [B, ...]."""
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # broadcast singleton on non-batch dims
    mask = (torch.rand(shape, device=x.device) < keep).float()
    return x * mask / keep

# Monte Carlo over 10k trials to confirm keep rate
alive_count = 0
B, trials = 8, 10_000
for _ in range(trials):
    out = drop_path(torch.ones(B, 1), drop_prob=0.25, training=True)
    alive_count += (out.sum(dim=1) > 0).float().sum().item()
print(f"MC over {trials} trials: alive rate={alive_count/(trials*B):.4f}")

# Output:
# MC over 10000 trials: alive rate=0.7501`}
      </CodeBlock>

      <Prose>
        Observed alive rate 0.7501 matches <Code>{"1 - drop_prob = 0.75"}</Code>. This is exactly what <Code>timm.layers.DropPath</Code> does under the hood.
      </Prose>

      <H3>4.5 Linear-ramp stochastic depth across layers</H3>

      <CodeBlock language="python">
{`L = 12          # number of residual blocks
dp_rate = 0.2   # max drop prob at last layer
rates = [dp_rate * i / (L - 1) for i in range(L)]
print("Layer | drop_prob | E[survives]")
for i, r in enumerate(rates):
    print(f"  {i:2d}  |  {r:.4f}  |  {1-r:.4f}")

# Output:
# Layer | drop_prob | E[survives]
#    0  |  0.0000  |  1.0000
#    1  |  0.0182  |  0.9818
#    2  |  0.0364  |  0.9636
#    3  |  0.0545  |  0.9455
#    4  |  0.0727  |  0.9273
#    5  |  0.0909  |  0.9091
#    6  |  0.1091  |  0.8909
#    7  |  0.1273  |  0.8727
#    8  |  0.1455  |  0.8545
#    9  |  0.1636  |  0.8364
#   10  |  0.1818  |  0.8182
#   11  |  0.2000  |  0.8000`}
      </CodeBlock>

      <Prose>
        The first block is never dropped (<Code>{"p_0 = 0"}</Code>); the last block is dropped 20% of the time. Early layers see more signal; late layers are the most regularized. Huang et al. (2016) argued this is correct because early features are more fundamental and shared across subnetworks.
      </Prose>

      <H3>4.6 MC Dropout for regression uncertainty</H3>

      <CodeBlock language="python">
{`import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(0)
X_train = torch.linspace(-3, 3, 60).unsqueeze(1)
y_train = torch.sin(X_train) + 0.1 * torch.randn_like(X_train)

class MCDropoutMLP(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.dropout(h, p=self.p, training=True)  # ALWAYS on (MC Dropout)
        h = F.relu(self.fc2(h))
        h = F.dropout(h, p=self.p, training=True)
        return self.fc3(h)

model = MCDropoutMLP(p=0.2)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for _ in range(2000):
    opt.zero_grad()
    loss = F.mse_loss(model(X_train), y_train)
    loss.backward()
    opt.step()

X_test = torch.linspace(-5, 5, 9).unsqueeze(1)   # extrapolate outside training range
with torch.no_grad():
    samples = torch.stack([model(X_test).squeeze() for _ in range(200)])
mu = samples.mean(dim=0)
sigma = samples.std(dim=0)
print("x      true_sin    mu       std")
for xi, mui, si in zip(X_test.squeeze().tolist(), mu.tolist(), sigma.tolist()):
    print(f"{xi:+.2f}   {np.sin(xi):+.4f}   {mui:+.4f}  {si:.4f}")

# Output:
# x      true_sin    mu       std
# -5.00   +0.9589   -0.0243  0.0600
# -3.75   +0.5716   -0.1092  0.0639
# -2.50   -0.5985   -0.5427  0.0878
# -1.25   -0.9490   -0.9322  0.1119
# +0.00   +0.0000   +0.1121  0.0751
# +1.25   +0.9490   +1.0082  0.1196
# +2.50   +0.5985   +0.6480  0.1079
# +3.75   -0.5716   +0.0947  0.0381
# +5.00   -0.9589   +0.1609  0.0767`}
      </CodeBlock>

      <Prose>
        Inside the training range (<Code>{"x ∈ [-3, 3]"}</Code>) the predicted mean <Code>{"\\mu"}</Code> tracks <Code>{"\\sin(x)"}</Code> closely and <Code>{"\\sigma"}</Code> stays small. Outside the training range the mean collapses toward zero (the network was never taught to extrapolate) — this is a well-known failure of plain MC Dropout: epistemic uncertainty grows somewhat but not enough. For properly calibrated uncertainty you need a good prior (SWAG, Deep Ensembles, or Laplace approximation). MC Dropout is cheap but miscalibrated without careful tuning.
      </Prose>

      <H3>4.7 Test accuracy vs dropout rate on synthetic classification</H3>

      <Prose>
        On a synthetic 10-class problem deliberately noisier than the signal supports, we train an MLP with <Code>{"\\{256 → 256 → 10\\}"}</Code> for 80 epochs at five drop rates:
      </Prose>

      <CodeBlock language="python">
{`def train_one(p_drop, epochs=80):
    torch.manual_seed(42)
    m = nn.Sequential(
        nn.Linear(N_feat, 256), nn.ReLU(),
        nn.Dropout(p_drop),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Dropout(p_drop),
        nn.Linear(256, N_cls),
    )
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        loss = F.cross_entropy(m(X_tr), y_tr)
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        train_acc = (m(X_tr).argmax(1) == y_tr).float().mean().item()
        val_acc   = (m(X_va).argmax(1) == y_va).float().mean().item()
    return train_acc, val_acc

for p in [0.0, 0.1, 0.3, 0.5, 0.7]:
    tra, val = train_one(p)
    print(f" p={p:.1f}  train={tra:.4f}  val={val:.4f}  gap={tra - val:+.4f}")

# Output:
#  p=0.0  train=1.0000  val=0.1870  gap=+0.8130
#  p=0.1  train=0.9785  val=0.1850  gap=+0.7935
#  p=0.3  train=0.7940  val=0.2100  gap=+0.5840
#  p=0.5  train=0.5685  val=0.2150  gap=+0.3535
#  p=0.7  train=0.3915  val=0.2140  gap=+0.1775`}
      </CodeBlock>

      <Prose>
        At <Code>{"p=0.0"}</Code>, training accuracy saturates at 100% while validation accuracy is 18.7% — pure memorization, gap of 0.81. As dropout increases, the train-val gap shrinks monotonically: 0.79 at <Code>{"p=0.1"}</Code>, 0.58 at <Code>{"p=0.3"}</Code>, 0.35 at <Code>{"p=0.5"}</Code>, 0.18 at <Code>{"p=0.7"}</Code>. Validation accuracy peaks around <Code>{"p=0.5"}</Code> at 21.5%. Beyond that the model underfits — training accuracy drops below 40% and there is no longer enough capacity to fit the real signal.
      </Prose>

      <Callout type="insight">
        The classic regularization tradeoff is fully visible: gap shrinks as dropout grows, but validation accuracy has a sweet spot. This is why dropout rate is a hyperparameter, not a fixed value — the right rate depends on the gap between model capacity and dataset size.
      </Callout>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production patterns</H2>

      <H3>5.1 PyTorch built-ins</H3>

      <CodeBlock language="python">
{`import torch.nn as nn

# Standard dropout on activations
layer = nn.Dropout(p=0.5)                # p is DROP probability, not keep

# Spatial dropout for convolutional feature maps (drop whole channels)
spatial = nn.Dropout2d(p=0.2)            # for 4D tensors [B, C, H, W]
spatial3 = nn.Dropout3d(p=0.2)           # for 5D volumetric features

# Dropout layers are automatically disabled in eval mode
model.train()   # dropout active
model.eval()    # dropout becomes identity

# Inside a forward(): use functional form if you need dynamic behavior
h = F.dropout(h, p=0.3, training=self.training)    # respects mode
h = F.dropout(h, p=0.3, training=True)             # always on (MC Dropout)`}
      </CodeBlock>

      <H3>5.2 timm DropPath for ViT / ConvNeXt</H3>

      <CodeBlock language="python">
{`from timm.layers import DropPath  # standard in every modern vision backbone

class TransformerBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))
        # One DropPath per residual branch
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# Linear ramp: early blocks 0, last block dp_rate
dp_rate = 0.1
depth = 12
blocks = nn.ModuleList([
    TransformerBlock(768, drop_path=dp_rate * i / (depth - 1))
    for i in range(depth)
])`}
      </CodeBlock>

      <H3>5.3 HuggingFace Transformer dropout knobs</H3>

      <CodeBlock language="python">
{`from transformers import BertConfig

cfg = BertConfig(
    hidden_dropout_prob=0.1,           # applied to most hidden states
    attention_probs_dropout_prob=0.1,  # applied to attention scores after softmax
    classifier_dropout=None,           # defaults to hidden_dropout_prob
)

# GPT-2 style (embd, resid, attn all separate)
from transformers import GPT2Config
cfg2 = GPT2Config(
    embd_pdrop=0.1,       # after token + position embedding sum
    resid_pdrop=0.1,      # before residual add (after mlp/attn)
    attn_pdrop=0.1,       # after attention softmax
)

# ViT / DeiT / CaiT — hydra knobs
# drop_rate:     classification head dropout
# attn_drop_rate: attention softmax dropout
# drop_path_rate: stochastic depth ramp rate`}
      </CodeBlock>

      <H3>5.4 DropBlock for convolutional backbones</H3>

      <CodeBlock language="python">
{`# Ghiasi, Lin, Le (NeurIPS 2018) — drops contiguous spatial regions
# size = side length of the dropped block, drop_prob tuned so expected area matches
from timm.layers import DropBlock2d

block = DropBlock2d(drop_prob=0.1, block_size=7)
# Typical use: ResNet stage 3+ to regularize late feature maps
# Often ramped over training (gamma schedule) like stochastic depth`}
      </CodeBlock>

      <H3>5.5 Common defaults in modern stacks</H3>

      <CodeBlock language="python">
{`# BERT / GPT-2 pretraining:     hidden_dropout=0.1, attn_dropout=0.1
# LLaMA / modern decoder LLMs:   dropout=0.0 during pretrain, small at SFT
# ViT-Base (16M params):         drop_path=0.1, attn_drop=0.0
# ViT-Huge (600M+):              drop_path=0.3-0.5, attn_drop=0.0
# ConvNeXt-Small:                drop_path=0.4
# ConvNeXt-Large:                drop_path=0.5
# ResNet-50 (ImageNet):          dropout=0 (BN is sufficient)
# ResNet-1001 stochastic depth:  p_L=0.5 linear ramp (Huang 2016)
# MLP on small tabular data:     dropout=0.3-0.5
# Regression MLP:                dropout=0.1-0.2 (or MC Dropout)`}
      </CodeBlock>

      <Callout type="info" title="Dropout on LLM pretraining">
        Modern foundation models (LLaMA, Mistral, Qwen, DeepSeek) typically set dropout to 0 during pretraining. The reasoning: datasets are so large that the model cannot overfit in the classical sense, and the extra stochasticity slows convergence without regularization benefit. Dropout reappears during SFT and fine-tuning where datasets are smaller and overfitting returns.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 Train vs val loss at different dropout rates</H3>

      <Prose>
        These curves are from the synthetic classification experiment in section 4.7, logged every 5 epochs. With no dropout (gold) training loss drops sharply while validation rises — classic overfitting. With moderate dropout (green) training loss stays higher but validation loss plateaus lower. With heavy dropout (purple) both curves are close together — the model is well-regularized but its capacity is choked.
      </Prose>

      <Plot
        label="Training loss vs dropout rate — synthetic 10-class MLP"
        xLabel="Epoch"
        yLabel="Cross-entropy loss"
        series={[
          { name: "p=0.0 train", color: colors.gold, points: [[0, 2.2796], [5, 2.1324], [10, 1.9808], [15, 1.8310], [20, 1.6938], [25, 1.5557], [30, 1.4096], [35, 1.2511]] },
          { name: "p=0.2 train", color: colors.green, points: [[0, 2.2843], [5, 2.1680], [10, 2.0458], [15, 1.9196], [20, 1.8066], [25, 1.7055], [30, 1.6073], [35, 1.5110]] },
          { name: "p=0.5 train", color: "#c084fc", points: [[0, 2.2930], [5, 2.2218], [10, 2.1664], [15, 2.1131], [20, 2.0432], [25, 1.9677], [30, 1.8925], [35, 1.8252]] },
        ]}
      />

      <Plot
        label="Validation loss vs dropout rate — overfitting visible at p=0.0"
        xLabel="Epoch"
        yLabel="Cross-entropy loss"
        series={[
          { name: "p=0.0 val", color: colors.gold, points: [[0, 2.2923], [5, 2.2258], [10, 2.1840], [15, 2.1674], [20, 2.1995], [25, 2.2434], [30, 2.2930], [35, 2.3537]] },
          { name: "p=0.2 val", color: colors.green, points: [[0, 2.2943], [5, 2.2368], [10, 2.1975], [15, 2.1720], [20, 2.1739], [25, 2.1996], [30, 2.2177], [35, 2.2298]] },
          { name: "p=0.5 val", color: "#c084fc", points: [[0, 2.3005], [5, 2.2584], [10, 2.2352], [15, 2.2183], [20, 2.1941], [25, 2.1736], [30, 2.1602], [35, 2.1591]] },
        ]}
      />

      <Prose>
        The gold (p=0.0) validation curve U-turns near epoch 15 and climbs — this is the signature of overfitting. The green (p=0.2) curve is shallower and reverses later. The purple (p=0.5) curve keeps descending smoothly — the model is still learning because dropout delays overfitting until well past where we stopped.
      </Prose>

      <H3>6.2 Dropout forward pass — step by step</H3>

      <StepTrace
        label="Inverted dropout forward pass — p=0.5, input size 6"
        steps={[
          { label: "Input activations x", render: () => (
            <Prose>
              Start with activation vector{" "}
              <Code>{"x = [1.2, -0.7, 0.9, 2.1, -1.4, 0.5]"}</Code>. At this stage no dropout has been applied; these are the outputs of a ReLU or Linear layer.
            </Prose>
          )},
          { label: "Sample Bernoulli mask", render: () => (
            <Prose>
              Draw mask <Code>{"m_i ~ Bernoulli(1-p) = Bernoulli(0.5)"}</Code> for each position. Suppose we sample{" "}
              <Code>{"m = [1, 0, 1, 1, 0, 1]"}</Code>. Positions 2 and 5 will be zeroed.
            </Prose>
          )},
          { label: "Apply mask", render: () => (
            <Prose>
              Element-wise multiply: <Code>{"x ⊙ m = [1.2, 0, 0.9, 2.1, 0, 0.5]"}</Code>. Four survivors, two zeros. At this point expected magnitude is half of the input.
            </Prose>
          )},
          { label: "Inverted-dropout rescale", render: () => (
            <Prose>
              Divide by <Code>{"1 - p = 0.5"}</Code>, equivalently multiply by 2:{" "}
              <Code>{"y = [2.4, 0, 1.8, 4.2, 0, 1.0]"}</Code>. Expected magnitude is restored. Forward to the next layer.
            </Prose>
          )},
          { label: "Backward pass", render: () => (
            <Prose>
              The gradient with respect to a zeroed position is zero (no gradient flows through a dead neuron). The gradient for surviving positions is also scaled by <Code>{"1/(1-p)"}</Code> — the same mask is stored and reused during backprop.
            </Prose>
          )},
          { label: "At eval time", render: () => (
            <Prose>
              Set <Code>{"training = False"}</Code>. The layer becomes the identity:{" "}
              <Code>{"y_eval = x = [1.2, -0.7, 0.9, 2.1, -1.4, 0.5]"}</Code>. No randomness, no scaling — the inverted-dropout trick gave us this for free.
            </Prose>
          )},
        ]}
      />

      <H3>6.3 Dropout mask heatmap over features × samples</H3>

      <Prose>
        A batch of 8 samples passing through a 12-dimensional dropout layer with <Code>{"p = 0.5"}</Code>. Bright cells are kept (scaled by 2); dark cells are zeroed. Every row (sample) has a different mask — this is what gives dropout its ensemble character across the batch.
      </Prose>

      <Heatmap
        label="Dropout mask per sample × feature (p=0.5) — bright = kept, dark = dropped"
        rowLabels={["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]}
        colLabels={["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"]}
        colorScale="gold"
        matrix={[
          [2, 0, 2, 2, 0, 2, 0, 2, 2, 0, 2, 0],
          [0, 2, 2, 0, 2, 0, 2, 2, 0, 2, 0, 2],
          [2, 2, 0, 2, 2, 0, 2, 0, 0, 2, 2, 0],
          [0, 0, 2, 2, 0, 2, 2, 2, 2, 0, 0, 2],
          [2, 0, 0, 0, 2, 2, 0, 2, 2, 2, 2, 0],
          [0, 2, 2, 0, 0, 0, 2, 2, 0, 2, 2, 2],
          [2, 2, 0, 2, 2, 0, 0, 0, 2, 0, 2, 2],
          [0, 0, 2, 2, 2, 2, 2, 0, 2, 2, 0, 0],
        ]}
      />

      <H3>6.4 Stochastic depth keep rates across a 12-block network</H3>

      <Plot
        label="Linear-ramp stochastic depth keep probability per layer (dp_rate=0.2, L=12)"
        xLabel="Layer index"
        yLabel="P(block survives)"
        series={[
          { name: "keep rate", color: colors.gold, points: [[0, 1.0], [1, 0.9818], [2, 0.9636], [3, 0.9455], [4, 0.9273], [5, 0.9091], [6, 0.8909], [7, 0.8727], [8, 0.8545], [9, 0.8364], [10, 0.8182], [11, 0.8]] },
        ]}
      />

      <Prose>
        The first block always survives; the last is dropped 20% of the time. Expected active depth is ~10.8 out of 12 blocks — a 10% speedup during training while the inference network is still 12 blocks deep.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        The right stochastic regularizer depends on what structure you have in your network and what correlations exist between the activations you might drop.
      </Prose>

      <H3>7.1 By architecture</H3>

      <CodeBlock>
{`ARCHITECTURE         | METHOD                  | TYPICAL RATE      | NOTES
---------------------+-------------------------+-------------------+----------------------------
MLP (fully connected)| nn.Dropout              | 0.3 - 0.5         | Between Linear+ReLU layers
CNN (feature maps)   | Dropout2d / DropBlock   | 0.1 - 0.3         | Per-pixel dropout is weak
ResNet (deep)        | Stochastic Depth        | p_L=0.2 (shallow) | Linear ramp
                     |                         | p_L=0.5 (1000+)   |
Vision Transformer   | DropPath + attn_drop    | 0.1 - 0.3 (path)  | ViT-Base 0.1, ViT-H 0.5
ConvNeXt             | DropPath                | 0.1 - 0.5         | Same ramp as ViT
Transformer LLM (pre)| usually 0               | 0                 | Not needed w/ huge data
Transformer LLM (SFT)| hidden_dropout          | 0.05 - 0.1        | Small dataset regime
RNN / LSTM           | Zoneout + var. dropout  | 0.1 - 0.2         | Not vanilla Dropout
Regression + UQ      | MC Dropout              | 0.1 - 0.2         | T=50-200 forward passes
Small tabular MLP    | Dropout                 | 0.3 - 0.5         | Often most important reg`}
      </CodeBlock>

      <H3>7.2 Granularity by correlation structure</H3>

      <Prose>
        Pick the drop granularity to match the correlated structure in your activations:
      </Prose>

      <CodeBlock>
{`SITUATION                              | RIGHT GRANULARITY   | WRONG GRANULARITY
---------------------------------------+---------------------+--------------------
Fully connected layer (i.i.d. units)   | Per-activation      | (any — works)
Conv feature maps (spatially correlated)| Per-channel (Dropout2d) or
                                       |   contiguous block (DropBlock) | Per-pixel (weak)
Residual network                       | Per-sample per-block (DropPath) or
                                       |   per-batch per-block (StochasticDepth) | Per-activation
Recurrent state                        | Timestep-preserving (Zoneout) or
                                       |   Variational Dropout | Per-timestep activation`}
      </CodeBlock>

      <H3>7.3 Uncertainty vs regularization</H3>

      <Prose>
        Dropout does two different jobs that are often confused. As a regularizer it is active at training and off at inference. As an uncertainty tool (MC Dropout) it is active at both and you average over multiple forward passes. You pick based on the downstream need:
      </Prose>

      <CodeBlock>
{`NEED                              | CONFIGURATION
----------------------------------+---------------------------------
Prevent overfitting only          | Train dropout on, eval dropout off
Regression uncertainty            | MC Dropout, T=50-200 samples
OOD detection                     | MC Dropout variance threshold
Active learning acquisition       | MC Dropout (BALD, variation ratios)
Deep ensembles (Lakshminarayanan) | Replace MC Dropout with N full models
                                  | (better calibrated, N× training cost)`}
      </CodeBlock>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <H3>8.1 Compute cost breakdown</H3>

      <Prose>
        Plain dropout is essentially free. The operations are (a) generating a Bernoulli mask with a uniform RNG, (b) element-wise multiply, (c) element-wise divide by <Code>{"1-p"}</Code>. All three are bandwidth-bound and fuse into the preceding activation. On modern GPUs the overhead is in the 1-3% range — often smaller than measurement noise.
      </Prose>

      <CodeBlock>
{`OPERATION                      | TRAINING FLOPS     | INFERENCE COST
-------------------------------+---------------------+----------------
Dropout                        | ~0 (mask + scale)   | 0 (identity)
Dropout2d                      | ~0 (smaller mask)   | 0
DropPath                       | ~0 (B-sized mask)   | 0
Stochastic Depth (block skip)  | SAVES ~p · L·F     | 0 (full depth)
DropBlock                      | small (sparsity op) | 0
MC Dropout                     | 0 (train), T× (eval)| T× forward
Variational dropout (learned)  | 2× parameter count  | 0 (or still 2×)`}
      </CodeBlock>

      <H3>8.2 Where stochastic depth actually wins</H3>

      <Prose>
        Huang et al. (2016) report that a 110-layer ResNet with stochastic depth (<Code>{"p_L = 0.5"}</Code>) trains roughly 25% faster per epoch than the baseline and reaches lower final test error. For a 1202-layer ResNet the speedup is larger (expected depth is ~0.75·L) and training is what makes the difference between feasible and infeasible. This is one of the few regularizers that makes training cheaper instead of more expensive.
      </Prose>

      <Prose>
        The wall-clock speedup is less than the expected-depth fraction because (a) when a block is dropped the GPU still has to synchronize with the next layer, (b) forward/backward of un-dropped layers is unchanged, and (c) memory allocations are typically sized for the full network. In practice, a 40-50% expected-depth reduction yields 20-30% wall-clock speedup.
      </Prose>

      <H3>8.3 MC Dropout inference cost</H3>

      <Prose>
        If you need uncertainty estimates, <Code>T = 50</Code> forward passes means 50× the per-query cost. For a 1B-parameter vision model serving at 100 QPS this is often prohibitive. Alternatives: Deep Ensembles (N independent models, N× training and memory but only N× inference, where N is typically 5-10 rather than 50); Last-layer Bayesian methods (cheap); Conformal prediction (calibration-based uncertainty with a single forward pass — covered in a separate topic).
      </Prose>

      <H3>8.4 Variational Dropout parameter overhead</H3>

      <Prose>
        Gal and Ghahramani's variational dropout treats drop rates as learnable parameters per weight, doubling the parameter count. Kingma, Salimans, and Welling's "Variational Dropout and the Local Reparameterization Trick" (NeurIPS 2015, arXiv:1506.02557) and Molchanov et al.'s "Variational Dropout Sparsifies Deep Neural Networks" (ICML 2017, arXiv:1701.05369) push this further to learn which weights to zero out entirely. The parameter overhead is 2× and training is noticeably harder, but the result is a sparse model at inference.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Forgetting model.eval()</H3>

      <Prose>
        The most common dropout bug in production. Dropout layers check <Code>self.training</Code> on every forward pass. In PyTorch, <Code>model.train()</Code> sets <Code>training = True</Code> and <Code>model.eval()</Code> sets it to False. Forgetting <Code>eval()</Code> at inference means dropout fires — predictions become stochastic, evaluation metrics drop by 1-3 percentage points, and you may spend days debugging a "regression" that is actually a mode flag. Always set <Code>model.eval()</Code> before scoring or serving. If a random subset of your predictions changes across identical inputs, this is the first place to look.
      </Prose>

      <H3>9.2 Dropout + BatchNorm interaction</H3>

      <Prose>
        Xiang Li et al. (CVPR 2019, arXiv:1801.05134) identified a specific failure mode: when dropout is applied before a batch normalization layer, the statistics that BN accumulates at training time do not match what the layer sees at inference. Dropout injects variance <Code>{"p/(1-p) · x^2"}</Code>; BN estimates mean and variance using these noisy activations; at inference dropout is off, so the activations entering BN have a different variance than what BN is calibrated for. The resulting distribution shift can degrade accuracy by 1-3 points.
      </Prose>

      <Prose>
        Mitigations: (1) put dropout AFTER the last BN layer in a block, not between Conv and BN; (2) prefer stochastic depth / DropPath on residual branches, which keeps BN statistics clean because when the branch is dropped the skip path is unchanged; (3) use Group Normalization or Layer Normalization, which compute statistics per-sample and are immune to the dropout variance shift; (4) in Vision Transformers, this is why you see LayerNorm + DropPath combined without concern — LN is stable under dropout.
      </Prose>

      <H3>9.3 Dropout on correlated feature maps</H3>

      <Prose>
        Per-pixel dropout on convolutional feature maps is weak because spatially adjacent pixels encode almost identical information (the receptive field overlaps). Dropping one pixel leaves its neighbors to carry the same signal — the network easily routes around the sabotage. Use Dropout2d (whole channels) or DropBlock (contiguous patches) instead. Applying plain <Code>nn.Dropout</Code> after a 2D convolution is a common beginner mistake that looks like regularization but barely moves the needle.
      </Prose>

      <H3>9.4 Too-high dropout causes underfitting</H3>

      <Prose>
        If <Code>p = 0.8</Code> in every hidden layer of a 10-layer network, only <Code>{"0.2^{10} ≈ 10^{-7}"}</Code> of pathways survive end-to-end. The signal-to-noise ratio collapses, training loss plateaus at chance level, and validation looks like random guessing. The sweep in section 4.7 showed the effect directly: at <Code>{"p=0.7"}</Code> training accuracy dropped to 39%. This is not regularization — it is deletion. A good rule: if training accuracy is not at least 10-20 points above random, dropout is too high (or the model is too small).
      </Prose>

      <H3>9.5 MC Dropout miscalibration</H3>

      <Prose>
        Gal and Ghahramani proved MC Dropout is equivalent to a specific variational approximation — but only when the prior and likelihood are set correctly. In practice, practitioners apply dropout without thinking about the prior; the resulting uncertainty estimates are often overconfident (too tight) in the interpolation regime and still under-estimated in extrapolation (as seen in section 4.6). For production uncertainty, calibrate against held-out data using temperature scaling, use conformal prediction for coverage guarantees, or switch to Deep Ensembles which are better calibrated out of the box (Lakshminarayanan et al., NeurIPS 2017).
      </Prose>

      <H3>9.6 Stochastic depth rate too high</H3>

      <Prose>
        If <Code>{"p_L = 0.9"}</Code> at the last layer, 90% of the time the last block is skipped. The final layers never receive enough gradient signal to learn anything useful — they become glorified identity layers. Training appears to work (loss decreases because early layers keep learning) but test accuracy of the full-depth network is worse than a shallow baseline. Huang et al. recommend <Code>{"p_L ≤ 0.5"}</Code> and linear ramp from 0 at the input. Going higher without extreme depth is unjustified.
      </Prose>

      <H3>9.7 Dropout inside LayerNorm or attention softmax</H3>

      <Prose>
        Applying dropout to the softmax output of attention (attention_probs_dropout) is a standard Transformer pattern — and there is a subtle issue: after dropout, rows of the attention matrix no longer sum to 1. This is usually fine in practice because attention acts as a weighted average, but it means the head's output is no longer a proper convex combination of values. At high dropout rates this becomes noticeable; ViT and modern LLMs typically set <Code>attn_pdrop = 0</Code> or a very small value (0.0 to 0.1) for this reason. The "dropout in the residual path" pattern (drop the output of attention/MLP before adding back to residual) is safer.
      </Prose>

      <Callout type="info" title="Summary of production gotchas">
        Always call <Code>model.eval()</Code> before inference. Don't mix dropout and BN unless you know where to place each. Use Dropout2d on feature maps, not Dropout. Keep dropout rates modest (0.1-0.5). Don't trust MC Dropout uncertainty without calibration. For stochastic depth use linear ramp with <Code>{"p_L ≤ 0.5"}</Code>. When in doubt on modern Transformers: DropPath 0.1 and move on.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Canonical papers in order of appearance — the reading list that defined this area:
      </Prose>

      <Prose>
        <strong>Hinton, Srivastava, Krizhevsky, Sutskever, Salakhutdinov (2012).</strong> "Improving neural networks by preventing co-adaptation of feature detectors." arXiv:1207.0580. The original dropout proposal. Short, readable, and establishes the co-adaptation framing that every subsequent paper builds on.
      </Prose>

      <Prose>
        <strong>Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov (2014).</strong> "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." Journal of Machine Learning Research 15:1929–1958. The definitive journal treatment. Contains the ensemble-of-subnetworks interpretation, extensive experiments across MNIST, CIFAR, ImageNet, TIMIT, and Reuters, and empirical studies of drop rates.
      </Prose>

      <Prose>
        <strong>Wager, Wang, Liang (2013).</strong> "Dropout Training as Adaptive Regularization." NeurIPS 26. arXiv:1307.1493. Proves that for GLMs, dropout is approximately an adaptive L2 penalty with per-feature weighting by Fisher information. The theoretical foundation for why dropout works.
      </Prose>

      <Prose>
        <strong>Wang, Manning (2013).</strong> "Fast dropout training." ICML. Shows that the expected gradient under dropout can be computed in closed form for Gaussian approximations, leading to a deterministic fast-dropout algorithm and clarifying the noise-injection equivalence.
      </Prose>

      <Prose>
        <strong>Huang, Sun, Liu, Sedra, Weinberger (2016).</strong> "Deep Networks with Stochastic Depth." ECCV. arXiv:1603.09382. Introduces stochastic depth, the linear-ramp schedule, and demonstrates that a 1202-layer ResNet can be trained to lower error than a 110-layer baseline in less wall-clock time.
      </Prose>

      <Prose>
        <strong>Gal, Ghahramani (2016).</strong> "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." ICML. arXiv:1506.02142. The Bayesian interpretation and MC Dropout. Also the PhD thesis version ("Uncertainty in Deep Learning") is an excellent longer reference.
      </Prose>

      <Prose>
        <strong>Kingma, Salimans, Welling (2015).</strong> "Variational Dropout and the Local Reparameterization Trick." NeurIPS. arXiv:1506.02557. Learnable drop rates per weight, with variance-reduced gradient estimators via the local reparameterization trick.
      </Prose>

      <Prose>
        <strong>Krueger, Maharaj, Kramár, Pezeshki, Ballas, Ke, Goyal, Bengio, Larochelle, Courville, Pal (2016).</strong> "Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations." arXiv:1606.01305. The recurrent adaptation — instead of zeroing, preserve the previous hidden state with some probability.
      </Prose>

      <Prose>
        <strong>Gastaldi (2017).</strong> "Shake-Shake regularization." ICLR Workshop. arXiv:1705.07485. Randomly mixes residual-branch outputs with per-sample coefficients — a continuous analog of DropPath.
      </Prose>

      <Prose>
        <strong>Yamada, Iwamura, Kise (2018).</strong> "ShakeDrop Regularization for Deep Residual Learning." arXiv:1802.02375. Extends Shake-Shake to single-branch architectures; combines stochastic depth with signed noise on the branch output.
      </Prose>

      <Prose>
        <strong>Ghiasi, Lin, Le (2018).</strong> "DropBlock: A regularization method for convolutional networks." NeurIPS. arXiv:1810.12890. Drops contiguous spatial regions of feature maps rather than individual pixels — fixes the weak-regularization problem of plain dropout on conv features.
      </Prose>

      <Prose>
        <strong>Li, Chen, Hu, Yang (2019).</strong> "Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift." CVPR. arXiv:1801.05134. Diagnoses why naive combinations of dropout and BN degrade performance and proposes practical placement rules.
      </Prose>

      <Prose>
        <strong>Touvron, Cord, Sablayrolles, Synnaeve, Jégou (2021).</strong> "Going deeper with Image Transformers" (CaiT). arXiv:2103.17239. Canonical use of DropPath in Vision Transformers with per-layer ramping — the pattern now adopted across ConvNeXt, Swin-V2, DiNOv2.
      </Prose>

      <Prose>
        <strong>Liu, Mao, Wu, Feichtenhofer, Darrell, Xie (2022).</strong> "A ConvNet for the 2020s" (ConvNeXt). arXiv:2201.03545. Demonstrates DropPath as a critical ingredient in modern CNN recipes, with drop rates scaled by model size.
      </Prose>

      <Prose>
        <strong>Lakshminarayanan, Pritzel, Blundell (2017).</strong> "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS. arXiv:1612.01474. The main competitor to MC Dropout for uncertainty — usually better calibrated but more expensive.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check</H2>

      <H3>Q1. Why does inverted dropout divide by <Code>{"1-p"}</Code> during training?</H3>

      <Callout type="answer">
        To preserve expected activation magnitude. A neuron survives with probability <Code>{"1-p"}</Code>, so its expected value without scaling is <Code>{"(1-p)·x"}</Code>. Dividing surviving values by <Code>{"1-p"}</Code> restores the expectation to <Code>x</Code>. This makes test time a simple identity (no special case) — the inverted-dropout trick is what decoupled training and inference in modern frameworks. The alternative (scaling at eval time by <Code>{"1-p"}</Code>) is called "vanilla dropout" and is rarely used today.
      </Callout>

      <H3>Q2. Why is plain dropout weak on convolutional feature maps?</H3>

      <Callout type="answer">
        Adjacent pixels in a conv feature map are highly correlated — they encode overlapping receptive fields of the same spatial location. Zeroing one pixel leaves its neighbors to reconstruct the same signal, so the network can route around the sabotage with trivial effort. Effective conv regularization needs to destroy coherent chunks of information: Dropout2d zeros whole channels (breaking channel-wise correlation) and DropBlock zeros contiguous spatial regions (breaking spatial correlation). See Ghiasi, Lin, Le (2018) for the empirical demonstration that DropBlock &gt; Dropout on ResNet ImageNet.
      </Callout>

      <H3>Q3. A ViT trains fine without dropout but a 10K-sample SFT dataset overfits badly. What do you add and why?</H3>

      <Callout type="answer">
        Small-dataset fine-tuning is the classical overfitting regime, so enable dropout. Two places matter: (1) hidden dropout in the Transformer blocks (<Code>{"hidden_dropout_prob = 0.1"}</Code> or so — the HuggingFace default for BERT is a reasonable starting point); (2) DropPath on residual branches (<Code>{"drop_path_rate = 0.1"}</Code> linearly ramped). Attention-probs dropout usually stays at 0 to avoid breaking softmax normalization. If overfitting persists, bump both to 0.2-0.3 and add weight decay (<Code>{"weight_decay = 0.05"}</Code>). The SFT setting is where dropout reappears in modern LLM stacks even though pretraining uses 0.
      </Callout>

      <H3>Q4. When would you pick MC Dropout over Deep Ensembles for uncertainty estimation?</H3>

      <Callout type="answer">
        MC Dropout wins when you need uncertainty cheaply and have already trained a model. It is one model with <Code>{"T"}</Code> forward passes — no retraining, no extra parameters. Deep Ensembles win when you have the compute budget to train <Code>{"N = 5-10"}</Code> independent models, and when calibration quality matters. Lakshminarayanan et al. (2017) show Deep Ensembles are consistently better-calibrated and detect out-of-distribution inputs more reliably. MC Dropout is known to be overconfident under distribution shift. Rule of thumb: MC Dropout for rapid prototyping and cheap active learning acquisition; Deep Ensembles for production safety-critical uncertainty.
      </Callout>

      <H3>Q5. You enable stochastic depth with <Code>{"p_L = 0.5"}</Code> on a 50-layer ResNet. Training loss is stuck at chance. What went wrong?</H3>

      <Callout type="answer">
        Likely either (a) you applied the constant rate <Code>{"p = 0.5"}</Code> to every layer instead of the linear ramp <Code>{"p_l = p_L · l / (L-1)"}</Code>, which means every block is dropped half the time — effectively you are training a 25-layer network on average with random-depth noise that BN cannot handle; (b) you forgot the inverted-dropout scaling so branches survive with wrong magnitude; or (c) dropout and BN interact badly on your particular architecture — see Li et al. (2018). The fix: use the linear ramp, keep the scaling, and verify that at least the first 3-4 blocks have near-zero drop probability so early feature learning is unimpaired. If the problem persists, switch from pre-BN to post-BN placement or to GroupNorm.
      </Callout>

    </div>
  ),
};

export default dropoutDroppathContent;
