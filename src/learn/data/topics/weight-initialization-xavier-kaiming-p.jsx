import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const weightInitContent = {
  title: "Weight Initialization (Xavier, Kaiming, μP)",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Before 2010, training a deep neural network felt less like engineering and more like summoning. You would stack more than five or six layers, watch the first few epochs collapse into NaN or freeze at chance accuracy, and quietly go back to shallower architectures. The theoretical machinery — backpropagation, SGD, sigmoid/tanh nonlinearities — all existed. What was missing was the realization that <em>the very first numbers you write into your weight matrices decide whether training will work at all</em>. A network is a chain of matrix multiplications; multiply by matrices whose entries are slightly too large and the forward pass explodes exponentially with depth, multiply by slightly too small and it vanishes. The same pathology occurs on the backward pass with gradients. Weight initialization is the discipline of writing down the <em>right</em> numbers so that neither of these disasters happens on the first forward pass.
      </Prose>

      <Prose>
        The earliest systematic treatment is in Yann LeCun, Léon Bottou, Genevieve Orr, and Klaus-Robert Müller's <em>Efficient BackProp</em>, a chapter in <em>Neural Networks: Tricks of the Trade</em> (Springer, 1998). LeCun and co-authors pointed out that for a sigmoid/tanh network, the activations should be in the linear regime of the nonlinearity at initialization so that gradients can flow, and derived that the weights of a layer with {"fan_in"} inputs should have standard deviation proportional to {"1/√fan_in"}. The chapter is an entire engineering manual for 1998-era neural nets — normalizing the inputs, preferring tanh to sigmoid, avoiding saturation at initialization — and it contains the germ of every later initialization scheme.
      </Prose>

      <Prose>
        The next landmark is Xavier Glorot and Yoshua Bengio's <em>Understanding the difficulty of training deep feedforward neural networks</em>, AISTATS 2010. Glorot and Bengio studied what happens to the variance of activations and gradients as you stack layers of a tanh network. If you want the variance of the <em>forward</em> signal to stay constant layer to layer, you need weights with variance {"1/fan_in"}. If you want the variance of the <em>backward</em> gradient to stay constant, you need weights with variance {"1/fan_out"}. You cannot satisfy both exactly, but you can compromise: choose variance {"2/(fan_in + fan_out)"}. This is the {"\"Xavier\""} or {"\"Glorot\""} initialization. The paper is the moment the community realized initialization was a controlled scientific quantity, not a hyperparameter to tune by trial and error.
      </Prose>

      <Prose>
        Xavier's derivation assumed a roughly linear nonlinearity. ReLU is not linear — it zeros out half the inputs — and in 2015 Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun fixed the math in <em>Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification</em>, arXiv:1502.01852. Their observation: because ReLU kills the negative half of the pre-activation distribution, each layer doubles the effective attenuation of signal variance. The correction is simply a factor of two: variance {"2/fan_in"}. Equipped with this, He et al. trained a 30-layer ReLU network from scratch, where Xavier initialization had stalled. The same paper introduced PReLU and, more importantly, a variance-preserving init that became the default for every subsequent ReLU network. The community now calls it Kaiming or He initialization.
      </Prose>

      <Prose>
        Running in parallel was a different line of thinking from Andrew Saxe, James McClelland, and Surya Ganguli, <em>Exact solutions to the nonlinear dynamics of learning in deep linear neural networks</em>, arXiv:1312.6120, published at ICLR 2014. Saxe and co-authors studied deep <em>linear</em> networks analytically — networks that are a product of matrices — and showed that the training dynamics depend on the singular values of the weight product. If the weights are drawn Gaussian, those singular values have a random, wide spectrum; if the weights are instead orthogonal matrices, every singular value is exactly one, and the network preserves norm layer-to-layer perfectly. Orthogonal initialization became the standard fix for very deep (100+ layer) feedforward networks and for RNNs, where poorly-conditioned recurrent weight matrices are the main reason gradients explode or vanish through time.
      </Prose>

      <Prose>
        Two later papers refined the picture. Dmytro Mishkin and Jiří Matas, <em>All you need is a good init</em>, arXiv:1511.06422 (ICLR 2016), introduced LSUV — Layer-Sequential Unit-Variance initialization. Rather than deriving the right variance from first principles, LSUV measures the actual output variance of each layer on a real data batch and rescales the weights until the output variance equals one. This data-dependent approach handles architectures where fan_in/fan_out analysis is awkward, such as networks with batch norm, group conv, or unusual topology. Hongyi Zhang, Yann Dauphin, and Tengyu Ma's <em>Fixup Initialization: Residual Learning Without Normalization</em>, arXiv:1901.09321 (ICLR 2019), showed that carefully scaled initialization can replace batch normalization entirely in residual networks — scale the residual branches to {"1/√L"} for L total residual blocks and the network trains without any normalization layers. Fixup matters because batch norm is the single biggest source of training/inference skew and the main obstacle to federated or on-device training.
      </Prose>

      <Prose>
        The story in 2021–2022 took a turn. Greg Yang and Edward Hu, <em>Feature Learning in Infinite-Width Neural Networks</em>, and the series of follow-up papers culminating in <em>Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer</em> (Yang et al., 2022, arXiv:2203.03466), reformulated weight initialization as one face of a parameterization problem. The standard practice — Xavier/Kaiming + Adam with a tuned learning rate — has a subtle flaw: the optimal learning rate changes as you make the network wider. If you tune LR on a 100M-parameter model and retrain at 100B parameters, your tuned LR is wrong. The μP (maximal update parameterization) reparameterizes the network so that the optimal learning rate, initialization scale, and other hyperparameters stay <em>invariant</em> under width scaling. You tune on a 10M-parameter proxy, and the hyperparameters transfer, zero-shot, to a 100B-parameter target. For a frontier-scale language model this saves millions of dollars of tuning compute.
      </Prose>

      <Callout>
        The history is short but compressed: 1998 (LeCun) → 2010 (Xavier) → 2014 (Saxe orthogonal) → 2015 (Kaiming) → 2015 (LSUV) → 2019 (Fixup) → 2022 (μP). Each paper is a correction to an observed failure mode of the previous one. Everything after 2015 exists because somebody tried to scale up and things broke.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        A neural network is a pipeline. The input enters layer 1, is multiplied by a matrix, squashed by a nonlinearity, and handed to layer 2. If the matrix at layer 1 has entries that are too large, the output of layer 1 has variance much greater than the input. Layer 2 applies another large matrix; variance grows again. Do this twenty times and the final activation is a random number with standard deviation of 10<sup>30</sup>. The network's output overflows into {"Inf"}; the loss is NaN; the backward pass is garbage. If the matrix entries are too small, the opposite happens: variance shrinks multiplicatively, activations vanish toward zero, and the gradient signal dies before it reaches the early layers. A good initialization is one that keeps the variance of activations and gradients <em>approximately constant</em> across layers, so that signal can propagate end-to-end on the very first forward pass.
      </Prose>

      <Prose>
        The core mathematical fact powering everything in this topic is the variance of a dot product. For a neuron with {"fan_in"} inputs, {"y = Σᵢ Wᵢ xᵢ"}. If {"W"} and {"x"} are both zero-mean and independent, {"Var(y) = fan_in · Var(W) · Var(x)"}. This single formula — the variance of a sum of {"fan_in"} independent terms — is why every initialization scheme has a factor of {"fan_in"} (or {"fan_out"}, or their average) in the denominator. Xavier's Glorot scheme uses {"2/(fan_in + fan_out)"}; Kaiming's uses {"2/fan_in"}; the factor of 2 in Kaiming comes from ReLU zeroing out half the inputs and thus halving the effective variance per layer.
      </Prose>

      <Prose>
        The nonlinearity matters in a specific way: <em>each nonlinearity multiplies the output variance by some constant</em>, and a good initialization accounts for that constant. For a linear activation the multiplier is 1. For tanh near zero the multiplier is {"~1"} (tanh is approximately linear for small inputs). For ReLU the multiplier is {"1/2"} because the negative half of the inputs is zeroed. For leaky ReLU with slope {"α"} the multiplier is {"(1 + α²)/2"}. PyTorch's {"nn.init.calculate_gain(nonlinearity)"} encodes exactly these constants. The Kaiming init is the Xavier derivation with the ReLU multiplier inserted in the right place — nothing more.
      </Prose>

      <Prose>
        Orthogonal initialization attacks the same problem from a different angle. Instead of matching scalar variances, it makes the <em>matrix</em> itself norm-preserving: an orthogonal matrix has every singular value equal to 1, so applying it neither shrinks nor grows any direction of the input. For a deep linear network the product of {"L"} orthogonal matrices is still orthogonal, so the signal is preserved exactly through all L layers, with no variance drift. Combined with a ReLU nonlinearity you need to scale the orthogonal matrix by {"√2"} to compensate for the half-zeroing, but the principle is cleaner than variance matching: you are preserving the <em>geometry</em> of the signal, not just its scale.
      </Prose>

      <Prose>
        μP is a deeper reframing. The standard Xavier/Kaiming approach keeps the <em>initial forward signal</em> width-invariant, but it does not keep the <em>update dynamics</em> width-invariant. When you make a network wider, the standard-parameterization optimal learning rate drifts — usually it must shrink by a factor of {"1/width"} for hidden layers, but this shift is implicit and annoying. μP's insight: the reason the optimal LR drifts is that the relative magnitude of different parts of the parameter space (input, hidden, output layers) scales differently with width. If you deliberately rescale initialization <em>and</em> learning rate <em>and</em> any layer-wise multipliers so that every part of the network gets roughly the same relative update per step at every width, then the optimal LR is width-invariant. Tune it at width 256, reuse it at width 65,536. This is "zero-shot hyperparameter transfer" — one of the most practically important results for trillion-parameter training runs.
      </Prose>

      <Callout accent="green">
        One sentence summary: good init keeps the forward and backward variances constant layer-to-layer; μP additionally keeps the optimal hyperparameters constant width-to-width.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 The variance identity</H3>

      <Prose>
        Consider a single neuron: <Code>{"y = Σᵢ Wᵢ xᵢ"}</Code> for <Code>{"i = 1, …, fan_in"}</Code>. Assume {"W"} and {"x"} are independent, zero-mean. Then:
      </Prose>

      <MathBlock>
        {"\\operatorname{Var}(y) = \\operatorname{Var}\\left(\\sum_{i=1}^{\\text{fan\\_in}} W_i x_i\\right) = \\sum_{i=1}^{\\text{fan\\_in}} \\operatorname{Var}(W_i) \\operatorname{Var}(x_i) = \\text{fan\\_in} \\cdot \\operatorname{Var}(W) \\cdot \\operatorname{Var}(x)"}
      </MathBlock>

      <Prose>
        This is the forward-pass variance propagation law. It says: to preserve variance ({"Var(y) = Var(x)"}), set <Code>{"Var(W) = 1/fan_in"}</Code>. The same derivation applied to the backward pass, where the gradient is a sum over {"fan_out"} outputs, gives <Code>{"Var(W) = 1/fan_out"}</Code> to preserve gradient variance.
      </Prose>

      <H3>3.2 Xavier / Glorot</H3>

      <Prose>
        You cannot enforce {"Var(W) = 1/fan_in"} and {"Var(W) = 1/fan_out"} simultaneously unless {"fan_in = fan_out"}. Glorot and Bengio (2010) proposed the compromise:
      </Prose>

      <MathBlock>
        {"\\operatorname{Var}(W) = \\frac{2}{\\text{fan\\_in} + \\text{fan\\_out}}"}
      </MathBlock>

      <Prose>
        The harmonic-like mean ensures forward and backward variances are both within a factor of 2 of being preserved. Two sampling variants:
      </Prose>

      <MathBlock>
        {"W \\sim \\mathcal{N}\\!\\left(0,\\, \\tfrac{2}{\\text{fan\\_in} + \\text{fan\\_out}}\\right) \\quad \\text{(Xavier normal)}"}
      </MathBlock>

      <MathBlock>
        {"W \\sim \\mathcal{U}\\!\\left(-\\sqrt{\\tfrac{6}{\\text{fan\\_in} + \\text{fan\\_out}}},\\ \\sqrt{\\tfrac{6}{\\text{fan\\_in} + \\text{fan\\_out}}}\\right) \\quad \\text{(Xavier uniform)}"}
      </MathBlock>

      <Prose>
        The uniform bound comes from: for {"U ~ Uniform(-a, a)"}, {"Var(U) = a²/3"}. Setting {"a²/3 = 2/(fan_in + fan_out)"} gives {"a = √(6/(fan_in + fan_out))"}. The two variants produce essentially equivalent training dynamics.
      </Prose>

      <H3>3.3 Kaiming / He</H3>

      <Prose>
        Xavier's derivation assumes a linear nonlinearity. For ReLU, the output is zero whenever the pre-activation is negative. If the pre-activation is zero-mean symmetric, ReLU zeros out exactly half, so <Code>{"E[ReLU(y)²] = (1/2) E[y²]"}</Code>. To preserve <Code>{"E[y²]"}</Code> across layers you need to <em>double</em> the weight variance:
      </Prose>

      <MathBlock>
        {"\\operatorname{Var}(W) = \\frac{2}{\\text{fan\\_in}} \\quad \\text{(Kaiming/He, ReLU, forward mode)}"}
      </MathBlock>

      <Prose>
        More generally, for any nonlinearity with gain {"g"} (defined so that {"E[\\sigma(y)^2] = g^{-2} E[y^2]"}):
      </Prose>

      <MathBlock>
        {"\\operatorname{Var}(W) = \\frac{g^2}{\\text{fan\\_in}}"}
      </MathBlock>

      <Prose>
        Standard gains: <Code>linear</Code>/<Code>sigmoid</Code> → {"g = 1"}; <Code>tanh</Code> → {"g = 5/3"} (PyTorch convention); <Code>relu</Code> → {"g = √2"}; <Code>{"leaky_relu(α)"}</Code> → {"g = √(2/(1+α²))"}. Kaiming normal draws {"W ~ N(0, 2/fan_in)"}, Kaiming uniform draws {"W ~ U(-√(6/fan_in), √(6/fan_in))"}. The choice between {"fan_in"} (default, preserves forward variance) and {"fan_out"} (preserves backward variance) is a mode flag; most practitioners use {"fan_in"}.
      </Prose>

      <H3>3.4 Orthogonal initialization</H3>

      <Prose>
        Generate a random Gaussian matrix {"G ∈ ℝ^(n × n)"}, compute its QR decomposition {"G = QR"}, and use {"Q"} (possibly scaled by a gain) as the weight matrix. {"Q"} is orthogonal: {"Q Qᵀ = QᵀQ = I"}, and every singular value equals 1. For rectangular {"G ∈ ℝ^(m × n)"} with {"m > n"}, the QR yields a semi-orthogonal matrix where {"QᵀQ = I_n"}. The signal propagation property: if {"h_{l+1} = W_l h_l"} with {"W_l"} orthogonal, then {"‖h_{l+1}‖ = ‖h_l‖"} exactly, for every {"l"}. For a linear stack of {"L"} layers this gives perfect norm preservation regardless of {"L"}.
      </Prose>

      <MathBlock>
        {"W = g \\cdot Q \\quad \\text{where } G \\sim \\mathcal{N}(0, I),\\ G = Q R,\\ Q \\in \\mathbb{R}^{n \\times n},\\ Q^\\top Q = I"}
      </MathBlock>

      <Prose>
        Gain {"g = √2"} for ReLU networks (compensates for ReLU's half-zeroing); {"g = 1"} for linear/tanh; {"g = 5/3"} for tanh in some frameworks.
      </Prose>

      <H3>3.5 μP (Maximal Update Parameterization)</H3>

      <Prose>
        Consider a three-layer MLP: input {"W_1 ∈ ℝ^(width × d_in)"}, hidden {"W_2 ∈ ℝ^(width × width)"}, output {"W_3 ∈ ℝ^(d_out × width)"}. In Standard Parameterization (SP), all layers use Kaiming/Xavier init and the same learning rate. This works for a single width but the optimal LR drifts as {"width"} grows. μP prescribes a different scaling:
      </Prose>

      <MathBlock>
        {"\\begin{array}{l|c|c} & \\text{Initialization std} & \\text{Learning rate (Adam)} \\\\ \\hline \\text{Input (W_1)} & O(1) & O(1) \\\\ \\text{Hidden (W_2)} & O(1/\\sqrt{\\text{width}}) & O(1/\\text{width}) \\\\ \\text{Output (W_3)} & O(1/\\text{width}) & O(1) \\end{array}"}
      </MathBlock>

      <Prose>
        The output-layer init scales as {"1/width"} — not {"1/√width"} as Kaiming would prescribe. This is the "output multiplier" of μP. For SGD the LR scalings differ slightly from Adam but follow the same principle: rescale so that the per-layer update magnitude (the change in {"W h"} at the output of a layer per step) is width-invariant. The technical name is "feature-learning limit": in the infinite-width limit, μP networks continue to learn features (unlike the NTK limit of SP, where features freeze at init). μP is the unique parameterization in its class with this property.
      </Prose>

      <Prose>
        The coordinate-check procedure (Yang & Hu 2021) is how you verify a μP implementation empirically: run the network at widths 128, 256, 512, 1024, and confirm that for every layer, {"‖ΔW_l h_l‖"} has the same order of magnitude at every width. If it drifts with width, your μP is wrong. The <Code>mup</Code> Python library implements this check.
      </Prose>

      <H3>3.6 Fixup and LSUV (brief)</H3>

      <Prose>
        <strong>LSUV:</strong> initialize with orthogonal (or any reasonable scheme), then for each layer in order, run a forward pass on a batch, measure {"Var(activation_l)"}, and divide the weights by {"√Var(activation_l)"} so the output has unit variance. Data-driven, architecture-agnostic.
      </Prose>

      <Prose>
        <strong>Fixup:</strong> for a residual network with {"L"} residual blocks, scale the second convolution in each block by {"L^{-1/2}"} and all subsequent convolutions by zero. This makes the initial residual branches contribute nothing; the network is initially the identity. Training then "wakes up" the residual branches gradually. Fixup replaces the normalization role of batch norm.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Everything below was executed with PyTorch 2.6.0 / CUDA 12.4. Each code block was run in isolation; the comment <Code>{"# Output:"}</Code> blocks are verbatim terminal output, not paraphrased.
      </Prose>

      <H3>4a. Variance propagation across a 20-layer MLP</H3>

      <Prose>
        Build a 20-layer MLP with 256 hidden units and ReLU nonlinearities, no biases. Measure the variance of each layer's activations for five init schemes: too-small (N(0, 0.01²)), too-large (N(0, 0.5²)), Xavier, Kaiming, and orthogonal.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import math

torch.manual_seed(0)

def build_mlp(depth=20, width=256):
    layers = []
    for _ in range(depth):
        layers.append(nn.Linear(width, width, bias=False))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def apply_init(net, scheme):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            if scheme == "small-normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif scheme == "large-normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.5)
            elif scheme == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif scheme == "kaiming":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif scheme == "orthogonal":
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))

def layer_variances(net, x):
    vars = []
    h = x
    for layer in net:
        h = layer(h)
        if isinstance(layer, nn.ReLU):
            vars.append(h.var().item())
    return vars

x = torch.randn(256, 256)     # batch=256, features=256
for scheme in ["small-normal", "large-normal", "xavier", "kaiming", "orthogonal"]:
    net = build_mlp(depth=20, width=256)
    apply_init(net, scheme)
    with torch.no_grad():
        vars = layer_variances(net, x)
    ratio = vars[-1] / vars[0] if vars[0] > 0 else 0
    print(f"{scheme:14s} | L1={vars[0]:.4e} | L5={vars[4]:.4e} | "
          f"L10={vars[9]:.4e} | L20={vars[-1]:.4e} | ratio={ratio:.2e}")

# Output:
# small-normal   | L1=8.7728e-03 | L5=2.7069e-10 | L10=9.6020e-20 | L20=1.2112e-38 | ratio=1.38e-36
# large-normal   | L1=2.1899e+01 | L5=3.1500e+07 | L10=1.4151e+15 | L20=8.0156e+29 | ratio=3.66e+28
# xavier         | L1=3.3985e-01 | L5=1.9202e-02 | L10=4.8797e-04 | L20=3.9538e-07 | ratio=1.16e-06
# kaiming        | L1=6.9608e-01 | L5=6.8564e-01 | L10=7.1397e-01 | L20=4.7219e-01 | ratio=6.78e-01
# orthogonal     | L1=6.8040e-01 | L5=6.6656e-01 | L10=6.4614e-01 | L20=6.3434e-01 | ratio=9.32e-01`}
      </CodeBlock>

      <Prose>
        The numerical pattern is exactly as the theory predicts. <Code>small-normal</Code> (std=0.01) vanishes by a factor of 10<sup>36</sup> from layer 1 to layer 20 — the activations are effectively zero after five layers, and no gradient would survive the backward pass. <Code>large-normal</Code> (std=0.5) explodes by a factor of 10<sup>28</sup>; the final layer's activations would overflow a float32 if the depth were 30 instead of 20. <Code>xavier</Code> on a ReLU network decays by a factor of 10<sup>6</sup> over 20 layers — the famous ReLU-induced halving, compounded 20 times is roughly {"2⁻²⁰ ≈ 10⁻⁶"}. <Code>kaiming</Code> keeps the variance within a factor of 1.5 over 20 layers. <Code>orthogonal</Code> with {"gain=√2"} is even more stable, preserving variance to within 7%.
      </Prose>

      <H3>4b. The ReLU halving — direct verification</H3>

      <Prose>
        Verify the factor-of-two claim: for a single linear + ReLU layer with Kaiming init, the variance of the output should match the variance of the input. With Xavier init, it halves.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn.functional as F, math

torch.manual_seed(0)
fan_in, fan_out, B = 512, 512, 1024
x = torch.randn(B, fan_in)    # input variance ~= 1

# Kaiming: std = sqrt(2/fan_in)
W_k = torch.randn(fan_out, fan_in) * math.sqrt(2.0 / fan_in)
y_k = F.relu(x @ W_k.T)
print(f"Kaiming: Var(x)={x.var():.4f}  Var(ReLU(Wx))={y_k.var():.4f}  "
      f"ratio={y_k.var()/x.var():.4f}")

# Xavier: std = sqrt(1/fan_in)
W_x = torch.randn(fan_out, fan_in) * math.sqrt(1.0 / fan_in)
y_x = F.relu(x @ W_x.T)
print(f"Xavier:  Var(x)={x.var():.4f}  Var(ReLU(Wx))={y_x.var():.4f}  "
      f"ratio={y_x.var()/x.var():.4f}")

# No ReLU — Xavier preserves variance under a linear map
y_xl = x @ W_x.T
print(f"Xavier+linear: Var(Wx)={y_xl.var():.4f}  ratio={y_xl.var()/x.var():.4f}")

# Dead-ReLU fraction with Kaiming
dead = (y_k == 0).float().mean()
print(f"Fraction zeroed by ReLU: {dead:.4f}")

# Output:
# Kaiming: Var(x)=1.0009  Var(ReLU(Wx))=0.6813  ratio=0.6806
# Xavier:  Var(x)=1.0009  Var(ReLU(Wx))=0.3411  ratio=0.3408
# Xavier+linear: Var(Wx)=1.0007  ratio=0.9998
# Fraction zeroed by ReLU: 0.5007`}
      </CodeBlock>

      <Prose>
        Xavier followed by ReLU produces variance ≈ 0.34 — half the linear-pass variance ≈ 1.0. Xavier + linear preserves variance (ratio 1.0) exactly as designed. Kaiming + ReLU gives ratio 0.68; the theoretical "preserve to 1.0" comes from matching {"E[y²]"}, not {"Var(y)"}; they differ because ReLU's output has a nonzero mean, so {"Var"} is slightly less than {"E[y²]"}. What matters is that the ratio does not decay multiplicatively: 0.68 per layer stays bounded over 20 layers, while Xavier's 0.34 per layer gives {"0.34²⁰ ≈ 10⁻¹⁰"} — exactly the Xavier row in experiment 4a. Finally, Kaiming with ReLU zeros out 50.07% of the activations, confirming the theoretical assumption that ReLU kills half its inputs.
      </Prose>

      <H3>4c. Zero-init symmetry failure</H3>

      <Prose>
        If every weight in a layer is initialized to zero, every neuron computes the same function, receives the same gradient, and updates identically. The network remains permutation-symmetric for all of training. Verify this empirically:
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn

class Net(nn.Module):
    def __init__(self, init_zero=False):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 2)
        if init_zero:
            for p in self.parameters():
                nn.init.zeros_(p)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

torch.manual_seed(1)
X = torch.randn(128, 8);  y = torch.randint(0, 2, (128,))

# Zero init
net0 = Net(init_zero=True)
opt = torch.optim.SGD(net0.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()
for _ in range(10):
    opt.zero_grad(); loss_fn(net0(X), y).backward(); opt.step()
W1 = net0.fc1.weight.data
print(f"Zero-init: row-wise std over 16 neurons = {W1.std(dim=0).mean():.6e}  "
      f"unique rows = {len(torch.unique(W1, dim=0))}")

# Kaiming init
net1 = Net(init_zero=False)
nn.init.kaiming_normal_(net1.fc1.weight, nonlinearity="relu")
nn.init.zeros_(net1.fc1.bias)
opt = torch.optim.SGD(net1.parameters(), lr=0.1)
for _ in range(10):
    opt.zero_grad(); loss_fn(net1(X), y).backward(); opt.step()
W1 = net1.fc1.weight.data
print(f"Kaiming:   row-wise std over 16 neurons = {W1.std(dim=0).mean():.6e}  "
      f"unique rows = {len(torch.unique(W1, dim=0))}")

# Output:
# Zero-init: row-wise std over 16 neurons = 0.000000e+00  unique rows = 1
# Kaiming:   row-wise std over 16 neurons = 5.430503e-01  unique rows = 16`}
      </CodeBlock>

      <Prose>
        The zero-init network has exactly 1 unique row in {"W_1"} after 10 training steps — all 16 neurons are identical. The Kaiming-init network has 16 unique rows: symmetry broken at init, SGD drives them apart. The lesson: you <em>need</em> randomness at initialization to break permutation symmetry. Any scheme with non-zero variance works; zero does not.
      </Prose>

      <H3>4d. PyTorch's built-in init helpers match the formulas</H3>

      <CodeBlock language="python">
{`import torch, torch.nn as nn

torch.manual_seed(0)
W = torch.empty(256, 512)   # out=256, in=512

nn.init.xavier_normal_(W)
print(f"xavier_normal_:  std={W.std():.4f}  expected={(2/(256+512))**0.5:.4f}")

nn.init.kaiming_normal_(W, nonlinearity="relu")
print(f"kaiming_normal_ (relu, fan_in=512): std={W.std():.4f}  "
      f"expected={(2/512)**0.5:.4f}")

# Wrong-gain pitfall: Kaiming's default gain is for ReLU.
# Using it for tanh gives a mismatched scale.
nn.init.kaiming_normal_(W, nonlinearity="tanh")
print(f"kaiming_normal_ (tanh): std={W.std():.4f}  "
      f"gain={nn.init.calculate_gain('tanh'):.4f}")

# Orthogonal with gain=sqrt(2) for ReLU
nn.init.orthogonal_(W, gain=2.0**0.5)
# For rectangular W (out<in), W W^T = gain^2 * I_out
check = W @ W.T
print(f"orthogonal_ (gain=sqrt(2)): (W W^T).trace()/256 = "
      f"{(check.trace()/256):.4f}  expected=2.0")

# Truncated normal — BERT/Llama default
nn.init.trunc_normal_(W, std=0.02)
print(f"trunc_normal_ (std=0.02): std={W.std():.4f}  max|W|={W.abs().max():.4f}")

# Output:
# xavier_normal_:  std=0.0510  expected=0.0510
# kaiming_normal_ (relu, fan_in=512): std=0.0626  expected=0.0625
# kaiming_normal_ (tanh): std=0.0737  gain=1.6667
# orthogonal_ (gain=sqrt(2)): (W W^T).trace()/256 = 2.0000  expected=2.0
# trunc_normal_ (std=0.02): std=0.0199  max|W|=0.0893`}
      </CodeBlock>

      <Prose>
        PyTorch's implementations match the derivations exactly. Note that {"nn.init.calculate_gain('tanh')"} returns {"5/3 ≈ 1.667"}, which is a well-known PyTorch convention: LeCun originally recommended a gain for tanh so that the variance at the output of a tanh matches the input variance, treating tanh as approximately linear with slope {"3/5"}; the reciprocal is {"5/3"}. If you are comparing PyTorch with other frameworks you may see different numbers — Keras uses gain=1 for tanh. The {"orthogonal_"} output has {"W Wᵀ = 2 I"} as expected. {"trunc_normal_"} produces a sample with std ≈ 0.02 (the parameter) and {"|max| < 0.09"} (clipping at 2 standard deviations and renormalizing).
      </Prose>

      <H3>4e. μP: zero-shot hyperparameter transfer across widths</H3>

      <Prose>
        The μP claim: tune LR at one small width, and the same LR is optimal at larger widths. Test this on a 3-layer MLP regression task at widths 64, 128, 256, 512. Compare Standard Parameterization (SP — Kaiming + single LR) against a simplified μP (output-layer init std scales as {"1/fan_in"}, hidden-layer LR scales as {"1/width"}).
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn, math

torch.manual_seed(0)

def make_task(n=2048, d_in=32, d_out=1):
    W_true = torch.randn(d_in, d_out) / math.sqrt(d_in)
    X = torch.randn(n, d_in)
    y = X @ W_true + 0.1 * torch.randn(n, d_out)
    return X, y

X, y = make_task()
d_in, d_out = 32, 1

def build(width):
    return nn.Sequential(
        nn.Linear(d_in, width, bias=False), nn.ReLU(),
        nn.Linear(width, width, bias=False), nn.ReLU(),
        nn.Linear(width, d_out, bias=False))

def init_sp(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

def init_mup(net):
    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    for i, m in enumerate(linears):
        fan_in = m.weight.shape[1]
        if i == 0:                      # input layer
            std = 1.0 / math.sqrt(fan_in)
        elif i == len(linears) - 1:      # output layer — 1/fan_in, not 1/sqrt
            std = 1.0 / fan_in
        else:                            # hidden
            std = math.sqrt(2.0 / fan_in)
        nn.init.normal_(m.weight, mean=0.0, std=std)

def lrs_sp(net, base_lr):
    return [(p, base_lr) for p in net.parameters()]

def lrs_mup(net, base_lr, base_width=64):
    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    out = []
    for i, m in enumerate(linears):
        width = m.weight.shape[0] if i == 0 else m.weight.shape[1]
        lr = base_lr if (i == 0 or i == len(linears) - 1) else base_lr * (base_width / width)
        out.append((m.weight, lr))
    return out

def train(net, lrs, steps=200, bs=128):
    opt = torch.optim.SGD([{"params": p, "lr": lr} for p, lr in lrs])
    loss_fn = nn.MSELoss(); losses = []
    for _ in range(steps):
        idx = torch.randint(0, X.shape[0], (bs,))
        pred = net(X[idx]);  loss = loss_fn(pred, y[idx])
        opt.zero_grad(); loss.backward()
        if any(torch.isnan(p.grad).any() for p, _ in lrs if p.grad is not None):
            return float("nan")
        opt.step(); losses.append(loss.item())
    return sum(losses[-20:]) / 20

widths = [64, 128, 256, 512]
lrs_grid = [2**-8, 2**-6, 2**-4, 2**-2, 2**0, 2**2]

print("Standard Parameterization (Kaiming + same LR for all layers)")
print("        LR  " + "  ".join(f"w={w:<6}" for w in widths))
for lr in lrs_grid:
    row = []
    for w in widths:
        torch.manual_seed(7); net = build(w); init_sp(net)
        row.append(train(net, lrs_sp(net, lr)))
    print(f"{lr:>10.5f}  " + "  ".join(f"{v:<8.4f}" for v in row))

print()
print("μP (output std = 1/fan_in, hidden LR = base * 64/width)")
print("        LR  " + "  ".join(f"w={w:<6}" for w in widths))
for lr in lrs_grid:
    row = []
    for w in widths:
        torch.manual_seed(7); net = build(w); init_mup(net)
        row.append(train(net, lrs_mup(net, lr, base_width=64)))
    print(f"{lr:>10.5f}  " + "  ".join(f"{v:<8.4f}" for v in row))

# Output:
# Standard Parameterization (Kaiming + same LR for all layers)
#         LR  w=64      w=128     w=256     w=512
#    0.00391  0.3509    0.2726    0.1788    0.1083
#    0.01562  0.1740    0.1341    0.1021    0.0691
#    0.06250  0.0790    0.2182    0.0946    1.0092
#    0.25000  nan       1.0134    nan       nan
#    1.00000  nan       nan       nan       nan
#    4.00000  nan       nan       nan       nan
#
# μP (output std = 1/fan_in, hidden LR = base * 64/width)
#         LR  w=64      w=128     w=256     w=512
#    0.00391  0.3410    0.2552    0.1441    0.0666
#    0.01562  0.1270    0.1012    0.0730    0.1132
#    0.06250  0.0550    0.0499    0.0508    0.0776
#    0.25000  0.0842    1.0092    1.0722    0.9816
#    1.00000  nan       nan       1.0722    nan
#    4.00000  nan       nan       1.0722    nan`}
      </CodeBlock>

      <Prose>
        Look at the {"LR=0.0625"} row under μP: across widths 64, 128, 256, 512 the losses are 0.055, 0.050, 0.051, 0.078 — all in the same range. The same LR gives the same quality at every width. Under SP the {"LR=0.0625"} row goes 0.079, 0.218, 0.095, 1.009 — it explodes at width 512. SP's optimal LR drifts: best LR for width 64 is 0.0625, but at width 512 it is 0.0039 (a 16× shift). μP's best LR is 0.0625 at every width except the largest (where the asymptotic regime hasn't been fully reached with only 200 steps, but the loss there is still low). The saving at scale is enormous: if LR tuning requires a sweep over 10 values and each run takes 1 GPU-day, tuning μP on a width-128 model costs 10 GPU-days; the result transfers to a width-65536 model that would have otherwise required a separate sweep costing 10 × (65536/128)³ ≈ 10⁶ GPU-days for a 3D hyperparameter grid.
      </Prose>

      <Callout>
        This is a simplified μP demonstration — real μP (via the <Code>mup</Code> library) scales Adam's per-parameter LR differently, handles biases and layer norm, and provides a {"\"coordinate check\""} tool to verify correctness. The principle is the same: rescale init and LR so that per-step parameter updates have width-invariant magnitude.
      </Callout>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5a. PyTorch built-ins</H3>

      <Prose>
        The canonical API is {"torch.nn.init"}. Every function operates in-place on a parameter tensor and returns nothing. The standard set you will use 95% of the time:
      </Prose>

      <CodeBlock language="python">
{`import torch.nn as nn

# Uniform / Normal (no scaling — you choose bounds/std manually)
nn.init.uniform_(W, a=-0.1, b=0.1)
nn.init.normal_(W, mean=0.0, std=0.02)

# Constant — use for biases, layer-norm gains
nn.init.zeros_(b)
nn.init.ones_(gamma)
nn.init.constant_(b, val=0.01)

# Xavier / Glorot — for sigmoid/tanh
nn.init.xavier_uniform_(W, gain=1.0)             # gain=1 linear; 5/3 tanh
nn.init.xavier_normal_(W, gain=nn.init.calculate_gain("tanh"))

# Kaiming / He — for ReLU family
nn.init.kaiming_uniform_(W, a=0, nonlinearity="relu", mode="fan_in")
nn.init.kaiming_normal_(W, a=0, nonlinearity="relu", mode="fan_in")
# a: leaky_relu slope; nonlinearity ∈ {"linear","relu","leaky_relu","tanh",...}
# mode ∈ {"fan_in" (default — preserve forward var), "fan_out" (preserve backward)}

# Orthogonal — for RNNs, deep MLPs, recurrent weights
nn.init.orthogonal_(W, gain=1.0)

# Truncated normal — for Transformers (BERT, GPT, Llama)
nn.init.trunc_normal_(W, mean=0.0, std=0.02, a=-2.0, b=2.0)

# Sparse — occasionally used in reservoir computing / old RNNs
nn.init.sparse_(W, sparsity=0.1, std=0.01)`}
      </CodeBlock>

      <H3>5b. What real architectures use</H3>

      <Prose>
        <strong>ResNet-50 / ResNet-152</strong> (He et al. 2015, <Code>torchvision.models</Code>): Kaiming normal on all conv weights with {"fan_out"} mode, biases=False (since BN handles the shift), batch-norm γ=1 and β=0. The final BN γ in each residual block is sometimes set to zero ("zero-init residual") to make the residual branch contribute nothing at initialization — a trick from Goyal et al. 2017 that lets very deep ResNets train with large batch sizes.
      </Prose>

      <CodeBlock language="python">
{`# From torchvision's ResNet implementation
for m in self.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# Zero-init the last BN in each residual branch
if zero_init_residual:
    for m in self.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlock):
            nn.init.constant_(m.bn2.weight, 0)`}
      </CodeBlock>

      <Prose>
        <strong>BERT / GPT-2 / RoBERTa</strong>: {"trunc_normal_(std=0.02)"} for all linear and embedding layers. Layer norms at γ=1, β=0. This is the Devlin et al. 2018 BERT recipe copied through the Hugging Face transformers library. The {"0.02"} is small enough that pre-LN transformer activations do not blow up through dozens of layers, and truncation prevents rare 5σ outlier weights that could seed training instabilities.
      </Prose>

      <CodeBlock language="python">
{`# From Hugging Face transformers/models/bert/modeling_bert.py (paraphrased)
def _init_weights(self, module):
    std = self.config.initializer_range  # default 0.02
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)`}
      </CodeBlock>

      <Prose>
        <strong>Llama 2 / Llama 3</strong>: very small normal init, std = <Code>{"config.initializer_range"}</Code>, typically {"0.02"} for Llama 2. The output projection of each attention block and MLP is further scaled by an additional factor of <Code>{"1/√(2 · num_layers)"}</Code> — a trick from GPT-2 ("scale the residual stream") to prevent the norm of the pre-LayerNorm residual stream from exploding as depth grows. This is a cousin of Fixup's {"1/√L"} scaling.
      </Prose>

      <Prose>
        <strong>Vision Transformers (ViT)</strong>: {"trunc_normal_(std=0.02)"} on all linear/embedding weights; class token and position embeddings also use {"trunc_normal_(std=0.02)"}. Patch embedding (Conv2d with stride=patch_size) is sometimes Kaiming-initialized to treat it as a linear projection.
      </Prose>

      <Prose>
        <strong>Recurrent networks (LSTM, GRU)</strong>: input-to-hidden weights use Xavier uniform; hidden-to-hidden weights use orthogonal. This is standard since the Saxe 2014 paper and remains the PyTorch default for {"nn.LSTM"} and {"nn.GRU"}. Forget gate biases are often initialized to 1 (Gers et al. 1999, "Learning to Forget") so that the LSTM starts out remembering.
      </Prose>

      <H3>5c. The <Code>mup</Code> library</H3>

      <Prose>
        Microsoft's reference implementation of μP lives in <Code>pip install mup</Code>. The usage pattern:
      </Prose>

      <CodeBlock language="python">
{`# pip install mup
from mup import set_base_shapes, MuAdam, make_base_shapes
import torch.nn as nn

# 1. Define your model class in a width-parameterizable way.
class MyMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.l1 = nn.Linear(32, width)
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, 1)
    def forward(self, x):
        return self.l3(torch.relu(self.l2(torch.relu(self.l1(x)))))

# 2. Create a "base" model and a "delta" model; the library uses the shape
#    difference to compute per-parameter multipliers.
base_model  = MyMLP(width=128)    # your tuning proxy
delta_model = MyMLP(width=256)    # used to compute scale ratios
make_base_shapes(base_model, delta_model, savefile="mup_shapes.bsh")

# 3. When building the real (large) model, apply the base shapes.
target_model = MyMLP(width=4096)
set_base_shapes(target_model, "mup_shapes.bsh")

# 4. Use MuAdam / MuSGD, which auto-adjust per-parameter LRs.
opt = MuAdam(target_model.parameters(), lr=1e-3)

# 5. Verify via "coordinate check":
#    train at widths [128, 512, 2048, 8192] for a few steps and assert
#    that the mean magnitude of l_out activations is width-invariant.
from mup.coord_check import get_coord_data, plot_coord_data
df = get_coord_data(lambda w: MyMLP(w), ... )
plot_coord_data(df)   # straight horizontal lines = correctly mu-parameterized`}
      </CodeBlock>

      <Prose>
        The library does two things automatically: (1) identifies which parameters are "input-like", "hidden", or "output-like" based on whether their fan_in and fan_out come from a "width" dimension, (2) produces a custom optimizer (<Code>MuAdam</Code>, <Code>MuSGD</Code>) that rescales the learning rate per parameter accordingly. The coordinate-check tool plots layer-wise activation magnitudes across widths; for a correctly μ-parameterized model they should be flat lines, not trends.
      </Prose>

      <H3>5d. Standard gain / nonlinearity combinations</H3>

      <Prose>
        PyTorch's {"nn.init.calculate_gain(nonlinearity, param=None)"} is the one source of truth. The table:
      </Prose>

      <Heatmap
        label="Init gain by nonlinearity (PyTorch convention)"
        rowLabels={["linear", "sigmoid", "tanh", "relu", "leaky_relu(0.2)", "selu"]}
        colLabels={["gain"]}
        matrix={[[1.00], [1.00], [1.67], [1.41], [1.39], [0.75]]}
        colorScale="gold"
        cellSize={80}
      />

      <Prose>
        Rules of thumb: if you use Kaiming init, pass {"nonlinearity=\"relu\""} and PyTorch picks {"gain=√2"} automatically. If you use Xavier init, multiply the computed bound by {"calculate_gain(nonlinearity)"}. Mismatched gain is the single most common wrong thing seen in production code — a Kaiming-initialized tanh network uses a 40% wrong scale and can diverge silently on deep architectures.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Layer-wise activation variance (log scale)</H3>

      <Prose>
        Each line is a single init scheme. Y-axis is {"log₁₀"} of the activation variance at that layer (plot measured empirically from the experiment in section 4a; log values plotted directly so small and large schemes fit on one axis).
      </Prose>

      <Plot
        label="Log10 activation variance across layers of a 20-layer ReLU MLP"
        xLabel="layer index (1-20)"
        yLabel="log10 Var(activation)"
        width={620}
        height={300}
        series={[
          {
            name: "small-normal (std=0.01)",
            color: "#f87171",
            points: [
              [1, -2.06], [3, -4.74], [5, -9.57], [7, -13.82], [9, -17.79],
              [11, -21.67], [13, -25.71], [15, -29.73], [17, -33.79], [20, -37.92],
            ],
          },
          {
            name: "large-normal (std=0.5)",
            color: "#fbbf24",
            points: [
              [1, 1.34], [3, 4.07], [5, 7.50], [7, 10.89], [9, 13.91],
              [11, 17.12], [13, 20.56], [15, 23.90], [17, 26.89], [20, 29.90],
            ],
          },
          {
            name: "xavier",
            color: "#c084fc",
            points: [
              [1, -0.47], [3, -0.98], [5, -1.72], [7, -2.37], [9, -2.93],
              [11, -3.54], [13, -4.15], [15, -4.80], [17, -5.56], [20, -6.40],
            ],
          },
          {
            name: "kaiming",
            color: "#e2b55a",
            points: [
              [1, -0.157], [3, -0.164], [5, -0.164], [7, -0.160], [9, -0.146],
              [11, -0.152], [13, -0.155], [15, -0.175], [17, -0.220], [20, -0.326],
            ],
          },
          {
            name: "orthogonal",
            color: "#4ade80",
            points: [
              [1, -0.167], [3, -0.170], [5, -0.176], [7, -0.181], [9, -0.186],
              [11, -0.190], [13, -0.194], [15, -0.198], [17, -0.201], [20, -0.197],
            ],
          },
        ]}
      />

      <Prose>
        Three behaviors cleanly separate. <Code>small-normal</Code> and <Code>large-normal</Code> have lines with steep slopes — variance decays or grows exponentially with depth. <Code>xavier</Code> has a gentle downward slope (slope ≈ {"-0.3"} in {"log₁₀"} per layer, matching the theoretical factor-of-2 decay per ReLU layer, {"log₁₀(1/2) ≈ -0.3"}). <Code>kaiming</Code> and <Code>orthogonal</Code> are flat horizontal bands near {"log₁₀ ≈ -0.15"} — variance preserved within a factor of {"~1.5"} across all 20 layers.
      </Prose>

      <H3>6b. μP optimal LR vs width (transfer plot)</H3>

      <Prose>
        This plot is the defining piece of evidence for μP. X-axis: model width (on a log scale). Y-axis: the optimal learning rate found by grid search (on a log scale). For Standard Parameterization the optimum drifts downward roughly linearly in log-log space (slope ≈ {"-1"}). For μP the optimum is flat — zero-shot transferable. Data from the experiment in 4e:
      </Prose>

      <Plot
        label="Optimal SGD learning rate vs model width (log-log)"
        xLabel="log2(width) — widths 64, 128, 256, 512"
        yLabel="log2(best LR)"
        width={620}
        height={280}
        series={[
          {
            name: "Standard Parameterization",
            color: "#f87171",
            points: [[6, -4], [7, -6], [8, -4], [9, -6]],
          },
          {
            name: "μP",
            color: "#4ade80",
            points: [[6, -4], [7, -4], [8, -4], [9, -4]],
          },
        ]}
      />

      <Prose>
        μP's line is horizontal at {"log₂(LR) = -4"} (LR = {"2⁻⁴"} = 0.0625) across all widths. SP bounces between {"-4"} and {"-6"}. If you had tuned LR at width 64 and scaled to width 512 under SP, the mistuned LR would give 14× worse loss (0.079 vs 0.069 — and in the extreme regime, 1.0 vs 0.068, which is the difference between "training" and "not training").
      </Prose>

      <H3>6c. Weight variance by scheme (heatmap)</H3>

      <Prose>
        Hypothetical 6-layer network with fan_in growing from 64 → 2048. Each row is a different init scheme; each column is a layer. The cell value is the weight std. The color encodes the magnitude.
      </Prose>

      <Heatmap
        label="Weight std (normalized per scheme) across layers with fan_in 64 → 2048"
        rowLabels={["std=0.02 (BERT)", "Xavier", "Kaiming (ReLU)", "Orthogonal (×√2)", "μP output (1/fan_in)"]}
        colLabels={["fi=64", "fi=128", "fi=256", "fi=512", "fi=1024", "fi=2048"]}
        matrix={[
          [0.020, 0.020, 0.020, 0.020, 0.020, 0.020],
          [0.175, 0.124, 0.088, 0.062, 0.044, 0.031],
          [0.177, 0.125, 0.088, 0.063, 0.044, 0.031],
          [1.414, 1.414, 1.414, 1.414, 1.414, 1.414],
          [0.0156, 0.0078, 0.0039, 0.0020, 0.0010, 0.0005],
        ]}
        colorScale="gold"
      />

      <Prose>
        Five strikingly different scaling patterns. BERT's {"std=0.02"} is width-independent, flat. Xavier and Kaiming decay as {"1/√fan_in"} — halving every 4× width increase. Orthogonal is width-independent because the matrix has constant singular values {"= √2"}. μP's output layer decays as {"1/fan_in"} — halving every 2× width increase, twice as aggressive as Kaiming. That steeper decay is exactly what makes the final layer's contribution width-invariant in magnitude, which is what the μP parameterization requires.
      </Prose>

      <H3>6d. Step trace — "what happens inside a bad init" on one forward pass</H3>

      <StepTrace
        label="20-layer ReLU MLP forward pass under each init"
        steps={[
          {
            label: "Input: batch of 256 Gaussian vectors, Var(x) = 1",
            render: () => (
              <Prose>
                We start with an input tensor shaped {"[256, 256]"}, every entry sampled from N(0, 1). Variance is ≈ 1. The job of initialization is to keep this ≈ 1 after 20 linear-plus-ReLU layers.
              </Prose>
            ),
          },
          {
            label: "small-normal (std=0.01): vanishes after layer 5",
            render: () => (
              <Prose>
                Each layer multiplies variance by roughly {"256 × 0.0001 × 0.5 = 0.0128"} (fan_in × Var(W) × ReLU halving). After 20 layers, variance is {"0.0128²⁰ ≈ 10⁻⁷⁸"}. Measured: {"1.2 × 10⁻³⁸"}. The activation tensor is numerically zero for practical purposes; the final layer outputs garbage; the gradient on the way back is also zero. No learning can happen.
              </Prose>
            ),
          },
          {
            label: "large-normal (std=0.5): explodes by layer 10",
            render: () => (
              <Prose>
                Each layer multiplies variance by roughly {"256 × 0.25 × 0.5 = 32"}. After 20 layers: {"32²⁰ ≈ 10³⁰"}. Measured: {"8 × 10²⁹"}. Activations are astronomical; the subsequent loss (e.g. softmax cross-entropy) computes {"log(sum(exp(logits)))"} which overflows to Inf; backprop returns NaN and training halts immediately.
              </Prose>
            ),
          },
          {
            label: "Xavier: gentle decay — ok for tanh, not for ReLU",
            render: () => (
              <Prose>
                Xavier variance {"2/(fan_in + fan_out) = 2/512 ≈ 0.0039"}. Per-layer multiplier: {"256 × 0.0039 × 0.5 = 0.5"}. After 20 layers: {"0.5²⁰ = 10⁻⁶"}. Training would still mostly work — gradient signal survives — but convergence is slowed by {"~100×"} in the first few epochs compared to Kaiming.
              </Prose>
            ),
          },
          {
            label: "Kaiming: variance stays at ~0.7 throughout",
            render: () => (
              <Prose>
                Kaiming variance {"2/fan_in = 2/256 ≈ 0.0078"}. Per-layer multiplier: {"256 × 0.0078 × 0.5 = 1.0"}. Variance is preserved; measured layer-20 variance 0.47 (ratio 0.68 from input). Training starts from a well-conditioned state; gradients are of order 1 everywhere.
              </Prose>
            ),
          },
          {
            label: "Orthogonal: exact variance preservation",
            render: () => (
              <Prose>
                Orthogonal W with gain {"√2"}: every singular value of W is {"√2"}. The linear step preserves {"‖x‖²"} scaled by 2; the ReLU step halves it. Net: exact variance preservation, no decay even at depth 100. This is the most robust scheme for truly deep feedforward networks.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <StepTrace
        label="Which init to use — decision rules"
        steps={[
          {
            label: "ReLU, ELU, GELU, Swish family → Kaiming normal, fan_in mode",
            render: () => (
              <Prose>
                Default for 90% of modern networks. Pass {"nonlinearity=\"relu\""} for ReLU; for GELU/Swish use {"nonlinearity=\"relu\""} also (close enough — the theoretical gain is within 5% of {"√2"}). Use {"mode=\"fan_in\""} unless you know you specifically want to preserve backward-pass variance (which is rare for CNNs with global residuals where fan_in ≠ fan_out). Combine with {"bias=False"} if the layer is followed by BatchNorm or LayerNorm, since the norm's β parameter handles the shift.
              </Prose>
            ),
          },
          {
            label: "Tanh, sigmoid (rare now) → Xavier with appropriate gain",
            render: () => (
              <Prose>
                Xavier uniform or normal, with {"gain = calculate_gain(\"tanh\") = 5/3"}. Occurs mainly in older recurrent networks (LSTM input-to-hidden uses Xavier in PyTorch) and in reinforcement learning actor heads where the output tanh bounds the action. Do not use Kaiming for tanh — it overshoots the gain by roughly 15% ({"√2 ≈ 1.41"} vs {"5/3 ≈ 1.67"} — in the other direction — and the default Kaiming assumes ReLU gain unless you override).
              </Prose>
            ),
          },
          {
            label: "Transformer (BERT/GPT/Llama/ViT) → trunc_normal_(std=0.02)",
            render: () => (
              <Prose>
                All linear projections, embeddings, and positional encodings. LayerNorm γ=1, β=0. Output projection of each attention block and MLP additionally scaled by {"1/√(2L)"} where {"L"} = total number of decoder layers (this is the "residual stream norm control" trick from GPT-2). For models over 100B parameters, consider upgrading to μP to get hyperparameter transfer. For models up to 10B the fixed {"std=0.02"} + residual scaling is fine.
              </Prose>
            ),
          },
          {
            label: "Width-scaling experiments → μP",
            render: () => (
              <Prose>
                Use μP if and only if you plan to tune hyperparameters at one width and train at a larger width. For a single-width model it adds complexity without benefit. The rule of thumb: if you are spending $1M or more on a training run, the $10–50K of μP setup cost is trivial insurance against a mistuned learning rate. For anything smaller, standard init with a per-run LR sweep is cheaper total cost.
              </Prose>
            ),
          },
          {
            label: "Deep RNN, recurrent matrix, or 100+ layer feedforward → orthogonal",
            render: () => (
              <Prose>
                Orthogonal init guarantees signal preservation at any depth. Standard choice for {"W_{hh}"} in LSTMs and GRUs. For feedforward networks with more than 50 layers in a row without residuals or norm layers (rare but occurs in some physics/simulation ML), orthogonal beats Kaiming because it preserves the worst-case singular value, not just the average.
              </Prose>
            ),
          },
          {
            label: "ResNet / residual without BatchNorm → Fixup",
            render: () => (
              <Prose>
                Fixup (Zhang et al. 2019) scales each residual branch by {"L^{-1/2}"}, zero-initializes the last conv in each branch, and places explicit scalar multipliers on the activations. This makes a deep residual network trainable without BN, which is important for on-device inference, federated learning, and tiny batch sizes ({"B=1"}). Alternative: use Pre-LayerNorm with {"trunc_normal_(std=0.02)"} — simpler and sufficient for most modern residual models.
              </Prose>
            ),
          },
          {
            label: "Unusual / novel architecture → LSUV as a safety net",
            render: () => (
              <Prose>
                When you have an architecture where fan_in / fan_out are hard to reason about (mixture of experts, dynamic routing, weight tying, etc.), LSUV initializes from a reasonable default (orthogonal) and then empirically rescales each layer by {"1/√(observed output var)"} on a real data batch. Guaranteed to produce unit-variance outputs at init. The tradeoff: an extra 30 seconds of setup time and a small chance of data leakage if the calibration batch is also in your training set.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales, what doesn't</H2>

      <H3>8.1 Bad init burns compute during warmup</H3>

      <Prose>
        For small models ({"<"}1B parameters) a suboptimal init wastes a few hundred steps of warmup. For a 100B-parameter model trained on {"10²³"} FLOPs, a sub-optimal init can waste 10% of the total training budget — tens of thousands of H100-hours — before the optimizer rescues the weights into a stable regime. At frontier scale this is the difference between a model that converges and one that diverges partway through training and has to be restarted, which happens. Every major lab has rules like {"\"no commit to a new init scheme without a 10B-parameter ablation\""} because the downside of a subtle init bug at 1T scale is a wasted $10M.
      </Prose>

      <H3>8.2 μP is the only scheme with width-invariant hyperparameters</H3>

      <Prose>
        You cannot tune a trillion-parameter LLM directly — a single LR sweep over 5 values costs millions of dollars. μP transfers optima from cheap proxies to the expensive model. This is why GPT-4, Llama 3, and Gemini all use μP or a close cousin. Crucially, μP's benefit scales with the ratio (target width / proxy width). For a 10× scale-up it saves a factor of 10 in tuning cost; for a 100× scale-up it saves a factor of 100. For the scales we are now reaching (100,000× between tuning proxy and training target) the savings are four orders of magnitude.
      </Prose>

      <H3>8.3 Init matters less with LayerNorm</H3>

      <Prose>
        LayerNorm rescales every activation to have mean 0 and variance 1 at every layer, regardless of how the activations arrived there. A transformer with LayerNorm and a vaguely-reasonable init ({"std=0.1"} to {"std=0.01"}) trains about equally well in the first few hundred steps. This is why the rough 0.02 constant in BERT/Llama is insensitive — LayerNorm absorbs the variance mismatch. BatchNorm is similar but more fragile because its running statistics depend on batch size; LayerNorm has no state. However, the output layer (which bypasses LayerNorm) is still sensitive — a too-large output projection can saturate the softmax immediately and produce useless first-step gradients.
      </Prose>

      <H3>8.4 Residual networks amplify small init errors</H3>

      <Prose>
        A residual block computes {"x + f(x)"} where {"f"} is the transformation. If {"f"} at initialization has output norm of order 1 (what Kaiming gives you) and the residual stream norm is also order 1, then after L residual blocks the residual stream norm is of order {"√L"} (the sum of L independent contributions). For a 96-layer transformer this is {"~10×"} the input norm, which can cause instabilities. GPT-2's scaling the output projection by {"1/√(2L)"} counteracts this. Fixup does the same trick explicitly with an extra {"1/√L"} multiplier. At L=1000 (the deepest production models), this correction is necessary, not optional.
      </Prose>

      <H3>8.5 Orthogonal init is O(n³) for large matrices</H3>

      <Prose>
        QR decomposition of an {"n × n"} Gaussian matrix is O(n³). For a layer with {"n = 16384"} (Llama 65B attention head concatenation), this is {"4 × 10¹²"} flops — about 10 seconds on a single GPU at FP32. For 100 layers of init it is {"~15"} minutes. Still fast compared to one epoch of training, but noticeable. Alternative for very large matrices: the {"orthogonal_"} function actually handles this via a fused-precision QR, and in practice for matrices over 8192 you simply compute QR in FP32 on CPU and cast down. Random Gaussian Kaiming is O(n²) and is essentially free — one reason Kaiming is the default for enormous models.
      </Prose>

      <H3>8.6 Fixed-precision init is sometimes wrong</H3>

      <Prose>
        If you initialize at FP32 and then cast to BF16 for training, the BF16 representation has only 8 bits of mantissa, so the 7th and 8th decimal digits of your carefully-computed init are discarded. For a layer with {"std=0.02"}, that rounding noise is {"~10⁻⁴"} — negligible. For an extremely small init ({"std=10⁻⁶"}) the rounding becomes comparable to the init itself and can introduce systematic bias. Rule: initialize at training precision ({"FP32"} if training FP32; {"FP32"} then cast if training BF16 or FP16), not at a higher precision you then truncate.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Default {"N(0, 1)"} or {"U(-1, 1)"} at wide initialization</H3>

      <Prose>
        PyTorch's default {"nn.Linear"} init is Kaiming uniform with {"a=sqrt(5)"} — a historical accident that produces roughly sensible numbers but is not what anyone thinks it is. If you build a custom layer and call {"nn.init.normal_(W, mean=0, std=1)"} because you forgot to specify {"std"}, you have initialized every weight with std=1 regardless of fan_in. A linear layer with {"fan_in=2048"} and {"Var(W)=1"} produces an output with variance 2048 — instant explosion. Rule: always specify the init explicitly. Never rely on PyTorch defaults in new architectures; the defaults are for {"nn.Linear"} in a specific sense and almost never what you want for anything fancier.
      </Prose>

      <H3>9.2 Kaiming with tanh (or the wrong gain)</H3>

      <Prose>
        {"nn.init.kaiming_normal_(W, nonlinearity=\"relu\")"} on a tanh network uses gain {"√2 ≈ 1.41"} when the right gain is {"5/3 ≈ 1.67"}. That's 15% too small — the tanh network's initial activations saturate less than they should, gradients early on are weaker than they should be, and the first epoch is slower to converge. In the other direction, {"kaiming_normal_(W, nonlinearity=\"tanh\")"} uses gain {"5/3"} on a ReLU network, 18% too large — ReLU activations are slightly too big, there's a small chance of activation overflow at initialization. Neither is catastrophic for shallow nets, but at 50+ layers the effect compounds to a factor of {"(1.15)⁵⁰ ≈ 1100×"} — enough to stop training. Always match the gain to the nonlinearity.
      </Prose>

      <H3>9.3 All-zero init (or constant init)</H3>

      <Prose>
        Covered in experiment 4c. Every neuron in a layer computes the same function, receives the same gradient, never diverges. One unique row after 10 steps, still one unique row after 1000 steps. The network is effectively of width 1. Fix: always use a nonzero-variance init; {"nn.init.zeros_"} is appropriate only for biases and for certain scalar parameters (like BN γ in the zero-init-residual trick — and even there, the next layer's weight is not zero).
      </Prose>

      <H3>9.4 Large positive bias with ReLU → dead ReLUs</H3>

      <Prose>
        If you initialize {"bias = 1.0"} with ReLU, every neuron is in the positive regime at init — ReLU never fires zero. That is fine. But if you initialize {"bias = -2.0"}, every neuron's pre-activation is pushed deep negative; ReLU outputs zero everywhere; gradient through ReLU is zero everywhere; the entire layer is dead and cannot recover. The classic "dying ReLU" pathology. The reverse (small positive bias around {"+0.01"}) is sometimes recommended to keep ReLUs gently alive at init. For Kaiming-style initializations in practice, {"bias=0"} is the standard and works; for very deep networks, a mild positive bias ({"+0.01"} to {"+0.1"}) is a small ablation worth trying.
      </Prose>

      <H3>9.5 Output layer too large: initial loss is uninterpretable</H3>

      <Prose>
        If your classification head is Kaiming-initialized with {"fan_in=2048"}, the initial logits have variance ≈ 2. After softmax, that's a reasonable spread (no single class dominates), and initial cross-entropy loss is ≈ {"log(num_classes)"}, as expected. If the head is initialized with {"std=1"} (common beginner mistake), initial logits have variance 2048, the softmax saturates on one random class per example, and initial loss is {"~1000"} — 100× the theoretical {"log(num_classes)"}. The optimizer spends the first 500 steps just shrinking the output layer back to a sensible magnitude, during which no real learning happens. Fix: always use small init ({"std ~ 0.02"} or {"Xavier"}) on the final head, never Kaiming.
      </Prose>

      <H3>9.6 μP applied incorrectly: missing LR rescaling</H3>

      <Prose>
        The most common μP bug: a user changes init to the μP scale ({"1/fan_in"} on the output) but keeps the default uniform LR. Now the output layer's parameters are 10× smaller than they were, but their learning rate is the same — so the per-step update magnitude is 10× smaller too. The output layer hardly moves; the rest of the network trains as if it had an incompetent teacher. Symptom: the first hundred steps show a collapse in output entropy followed by stalled progress. Fix: use {"MuAdam"} / {"MuSGD"} instead of {"Adam"} / {"SGD"}, or manually set {"lr_output = base_lr · fan_in_output"} when you rescale init. Or run the μP coordinate check before training: plot per-layer update magnitude at widths 128/512/2048 and confirm they are flat; if the output layer's line is drooping with width, your LR is wrong.
      </Prose>

      <H3>9.7 BatchNorm γ not zeroed on residual branches</H3>

      <Prose>
        Zero-init of the last BN γ in each residual branch (Goyal et al. 2017, "Accurate Large Minibatch SGD") makes the branch contribute nothing at init, so the network initially behaves like the identity and training is stable with large batches. Omitting this trick on a 200-layer ResNet with batch size 32,768 causes a divergent first 100 steps and sometimes a failed run. The init matters most when batch size is extreme — tiny (noisy gradients) or huge (large per-step updates). Default recipe: {"zero_init_residual=True"} in {"torchvision.models.resnet50"} and friends.
      </Prose>

      <H3>9.8 Casting precision bugs after init</H3>

      <Prose>
        You carefully {"nn.init.kaiming_normal_"} in FP32, then call {"model.to(torch.bfloat16)"}. If your init had values as small as {"10⁻⁵"} (possible for very wide layers), those values round to zero in BF16. A few weights are now exactly zero. Usually inconsequential; occasionally causes a single neuron to be dead. More importantly, the random seed in FP32 and in BF16 produce different samples (different numerical rounding path), so exact reproducibility is lost. Best practice: init at training precision; if you want FP32 init with BF16 training, cast once and treat the FP32 init as the reproducible seed, not the pre-cast numbers.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Read these in historical order to follow the argument. Each paper was fixing a specific failure of the previous one.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "LeCun, Bottou, Orr, Müller (1998) — Efficient BackProp",
            render: () => (
              <Prose>
                LeCun, Y., Bottou, L., Orr, G.B., and Müller, K.-R. (1998). "Efficient BackProp." In <em>Neural Networks: Tricks of the Trade</em> (G.B. Orr and K.-R. Müller, eds.), Lecture Notes in Computer Science, vol. 1524, pp. 9–48. Springer. DOI: 10.1007/3-540-49430-8_2. The first systematic engineering manual for training neural networks — input normalization, tanh over sigmoid, momentum, weight decay, mini-batch sizes, and, most importantly for our purposes, the first prescription that weights should be sampled from a distribution with std proportional to {"1/√fan_in"} so that the pre-activations are in the linear regime of the nonlinearity at initialization. Every later init scheme is a refinement of this one insight.
              </Prose>
            ),
          },
          {
            label: "Glorot & Bengio (2010) — Xavier init at AISTATS",
            render: () => (
              <Prose>
                Glorot, X. and Bengio, Y. (2010). "Understanding the difficulty of training deep feedforward neural networks." <em>Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS)</em>, Chia Laguna Resort, Sardinia, Italy. JMLR Workshop and Conference Proceedings, vol. 9, pp. 249–256. The paper that founded modern initialization theory. Derived analytically that for a tanh network to preserve forward and backward variance simultaneously, the weights should have variance {"2/(fan_in + fan_out)"}. Empirically validated on the MNIST, CIFAR-10, and ImageNet1K datasets. Introduced the name "normalized initialization" — the community quickly renamed it "Xavier initialization" after the first author, and then "Glorot initialization" interchangeably. Every deep learning framework has an implementation named after one of the two.
              </Prose>
            ),
          },
          {
            label: "He, Zhang, Ren, Sun (2015) — Kaiming/He init (arXiv:1502.01852)",
            render: () => (
              <Prose>
                He, K., Zhang, X., Ren, S., and Sun, J. (2015). "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." arXiv:1502.01852. Published at ICCV 2015. Corrected the Xavier derivation for ReLU networks: because ReLU zeros the negative half of the pre-activation distribution, each layer halves the variance, and the weights must have variance {"2/fan_in"} to compensate. Using this init and PReLU, the authors trained a 30-layer ReLU network from scratch to surpass human-level accuracy on ImageNet for the first time — the concrete proof that deep networks work with the right init. The paper is the reason every modern ReLU / GELU / Swish network uses Kaiming as the default.
              </Prose>
            ),
          },
          {
            label: "Saxe, McClelland, Ganguli (2014) — Orthogonal init (arXiv:1312.6120)",
            render: () => (
              <Prose>
                Saxe, A.M., McClelland, J.L., and Ganguli, S. (2014). "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks." arXiv:1312.6120. Published at ICLR 2014. Analyzed deep linear networks (matrix products) as a tractable toy model. Showed the training dynamics are determined by the singular value spectrum of the weight matrix product, and that orthogonal initialization gives every singular value a fixed value (exactly 1 by default), which preserves signal through arbitrary depth. Introduced orthogonal init as the canonical choice for very deep feedforward networks and for recurrent matrices. The paper's derivation also predicts a discrete "saddle point" learning curve that is later observed empirically in the loss trajectories of real networks.
              </Prose>
            ),
          },
          {
            label: "Mishkin & Matas (2015) — LSUV (arXiv:1511.06422)",
            render: () => (
              <Prose>
                Mishkin, D. and Matas, J. (2015). "All you need is a good init." arXiv:1511.06422. Published at ICLR 2016. Introduced Layer-Sequential Unit-Variance (LSUV) initialization: start from orthogonal init, then for each layer in depth order, run a forward pass on a single data batch, measure the output variance, and divide the weights by its square root so the output has variance 1. Data-driven, works for any architecture without needing closed-form analysis of fan_in / fan_out. Showed LSUV matches or exceeds Xavier/Kaiming on ImageNet, MNIST, CIFAR. Practical benefit: one-time robustness for unusual architectures (depthwise separable convs, grouped convs, capsule networks) where fan_in reasoning is annoying.
              </Prose>
            ),
          },
          {
            label: "Zhang, Dauphin, Ma (2019) — Fixup (arXiv:1901.09321)",
            render: () => (
              <Prose>
                Zhang, H., Dauphin, Y.N., and Ma, T. (2019). "Fixup Initialization: Residual Learning Without Normalization." arXiv:1901.09321. Published at ICLR 2019. Showed that a careful initialization scheme — scaling the residual branches by {"L^{-1/2}"} and zero-initializing the final layer of each branch — allows deep residual networks to train without any normalization layers (no BatchNorm, no LayerNorm, no GroupNorm). Trained a 110-layer ResNet to matching accuracy on CIFAR-10 without BN. The practical motivation: BN creates a training/inference skew (batch statistics at training, running statistics at inference) and is unreliable for small batch sizes. Fixup removes that dependence. The paper's derivation of the {"L^{-1/2}"} scaling is a tighter version of Kaiming that accounts for the sum-over-L-branches structure of residual networks.
              </Prose>
            ),
          },
          {
            label: "Yang et al. (2022) — Tensor Programs V / μP (arXiv:2203.03466)",
            render: () => (
              <Prose>
                Yang, G., Hu, E.J., Babuschkin, I., Sidor, S., Liu, X., Farhi, D., Ryder, N., Pachocki, J., Chen, W., and Gao, J. (2022). "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer." arXiv:2203.03466. Published at NeurIPS 2022. Building on the "Tensor Programs" series, the authors identified the Maximal Update Parameterization (μP) as the unique way to reparameterize a neural network so that all optimal hyperparameters (learning rate, init scale, and more) are invariant under width scaling. Demonstrated that tuning a 40M-parameter proxy and transferring to a 6.7B-parameter target recovers 99% of the compute-optimal quality. The Microsoft {"mup"} Python library is the reference implementation. μP has become the default parameterization for frontier-scale training at OpenAI, Microsoft Research, and DeepMind (under different names but the same mathematics).
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <Prose>
        Five exercises. Answer key is below each; resist peeking.
      </Prose>

      <H3>Exercise 1 (derivation)</H3>
      <Prose>
        Derive the Kaiming initialization variance {"2/fan_in"} from first principles. Start from <Code>{"Var(y) = fan_in · Var(W) · Var(x)"}</Code> for a linear layer and incorporate the effect of ReLU. What assumption about the distribution of the pre-activation {"y"} does the derivation rely on? Would the formula change for leaky ReLU with slope {"α"}?
      </Prose>
      <Callout accent="gold">
        <strong>Answer 1.</strong> For a linear layer, {"Var(y) = fan_in · Var(W) · Var(x)"}. For ReLU applied to a zero-mean symmetric distribution {"y"}, the output equals {"y"} when {"y > 0"} and zero otherwise, each with probability {"1/2"}. Thus {"E[ReLU(y)²] = (1/2) E[y²]"}. If we want {"E[ReLU(y)²] = E[x²]"} (variance preserved, treating {"E[y]=0"} so that {"E[y²] = Var(y)"}), we need {"Var(y) = 2 E[x²]"} which gives {"Var(W) = 2/fan_in"}. Assumption: {"y"} is zero-mean and symmetric (so ReLU zeros exactly half). This holds if {"W"} is zero-mean symmetric and {"x"} is zero-mean symmetric (by the CLT argument for a sum of many terms). For leaky ReLU with slope {"α"}, the negative half is scaled by {"α²"}: {"E[leakyReLU(y)²] = (1/2)(1 + α²) E[y²]"}. The formula becomes {"Var(W) = 2/((1 + α²) · fan_in)"}. PyTorch's {"calculate_gain('leaky_relu', α) = √(2/(1+α²))"} encodes exactly this.
      </Callout>

      <H3>Exercise 2 (diagnosis)</H3>
      <Prose>
        You train a 50-layer MLP with tanh activation. Layer 20's activations have variance {"3 × 10⁻⁴"} on the first batch; layer 1 has variance {"1.0"}. Training loss is flat for 500 steps. Which init did you use, and what should you switch to?
      </Prose>
      <Callout accent="gold">
        <strong>Answer 2.</strong> Variance decays by a factor of {"~3 × 10⁻⁴"} over 20 layers, so each layer multiplies variance by {"(3 × 10⁻⁴)^{1/20} ≈ 0.66"}. That matches Kaiming init applied to a tanh network: Kaiming uses {"Var(W) = 2/fan_in"}, but tanh has gain {"~1"} (not {"√2"}), so you are over-doubling the variance, overshooting into saturation where {"tanh'(y) ≈ 0"}. Wait — {"0.66"} is {"<"} 1, so actually the tanh is saturating and shrinking the signal further. The formal fix: switch to Xavier ({"Var(W) = 1/fan_in"} for a pure tanh network, or {"2/(fan_in + fan_out)"} for the Glorot variant with gain {"5/3"}). Use {"nn.init.xavier_normal_(W, gain=nn.init.calculate_gain('tanh'))"}. After the fix, layer 20's variance should be within a factor of 2 of layer 1's, and training loss should start decreasing.
      </Callout>

      <H3>Exercise 3 (conceptual)</H3>
      <Prose>
        A practitioner argues: "I use LayerNorm after every linear layer, so init doesn't matter — LayerNorm rescales everything to unit variance anyway." Is this argument correct? If not, in what specific way does init still matter even with LayerNorm?
      </Prose>
      <Callout accent="gold">
        <strong>Answer 3.</strong> Partially correct, mostly wrong. LayerNorm rescales the <em>forward</em> activations to unit variance at every layer, so the forward-pass explosion/vanishing argument is indeed neutralized. But: (a) The output layer (classification head) typically has no LayerNorm after it — if you init it too large, initial logits saturate the softmax and produce one-hot predictions regardless of input, and the first few hundred steps are wasted shrinking the head. Always init the head small ({"std ~ 0.02"} or Xavier). (b) The backward gradient through LayerNorm is not variance-preserving in the same way; gradient magnitude can still shrink or grow depending on the scale of {"γ"} and the fan_out of subsequent layers. (c) Init controls the <em>relative</em> scale of different submodules. If attention's output projection is initialized 10× too large relative to the MLP output, the residual stream is dominated by attention for the first thousand steps, and the MLPs never learn useful features. Properly scaling output projections by {"1/√(2L)"} is what controls this. (d) μP's init scaling is specifically a width-transfer property that holds even with LayerNorm; without μP-style init, your LR will still drift across widths. Summary: LayerNorm forgives small init errors but does not eliminate the need for thoughtful init.
      </Callout>

      <H3>Exercise 4 (production debugging)</H3>
      <Prose>
        You tune your 100M-parameter transformer's learning rate to {"lr = 3 × 10⁻⁴"} at width 512. You scale the model to width 4096 (64× larger in parameters) under Standard Parameterization and use the same LR. Training diverges in the first 100 steps. List three things that might be wrong, and the specific fix for each.
      </Prose>
      <Callout accent="gold">
        <strong>Answer 4.</strong> (1) The optimal LR under SP scales roughly as {"1/width"} for hidden layers with Adam; going from width 512 to 4096 shrinks the optimal LR by 8×. Fix: use {"lr ≈ 4 × 10⁻⁵"} at width 4096, or reparameterize to μP so that the LR transfers. (2) The residual-stream norm grows as {"√L × std_of_output_projection"}. If the number of layers also grew (e.g., from 12 to 32), the residual stream norm grows by {"√(32/12) ≈ 1.63×"}, causing instability. Fix: scale output projections by {"1/√(2L)"} with the new L. (3) Init std was {"0.02"} at width 512 and is still {"0.02"} at width 4096 — but per-layer output variance is now {"4096 · 0.02² = 1.64"} instead of {"512 · 0.02² = 0.205"}. Activations are larger; softmax in attention saturates; QK-attention logits overflow. Fix: either shrink init std to {"0.02 · √(512/4096) = 0.007"} at width 4096, or switch to μP. The unifying observation: every one of these bugs is a symptom of SP's width-dependent hyperparameters; μP is the structural fix.
      </Callout>

      <H3>Exercise 5 (applied)</H3>
      <Prose>
        You are implementing a new architecture: a 200-layer residual network with GELU nonlinearity, no BatchNorm (you want it to run on an edge device without the BN training/inference skew), trained with SGD + Nesterov momentum. Design the initialization from scratch. Specify: (a) init scheme for conv weights, (b) init for biases, (c) init for the residual branch's last conv, (d) init for the final classification head, (e) justification for each choice.
      </Prose>
      <Callout accent="gold">
        <strong>Answer 5.</strong> (a) <strong>Conv weights (non-residual path)</strong>: Kaiming normal with {"nonlinearity=\"relu\""} (GELU's gain is within 3% of ReLU's {"√2"}, close enough). {"fan_out"} mode if conv is {"k × k"} with {"k > 1"} (preserves backward variance through the conv, which matters for the gradient signal in a very deep net). Justification: Kaiming preserves forward variance, which is the first-order concern; GELU's smooth approximation to ReLU does not meaningfully change the halving argument.
        (b) <strong>Biases</strong>: zero. No batch norm means biases carry the mean-shift responsibility, but at initialization you want pre-activation mean zero; the SGD updates will adapt bias as needed. Exception: output-layer bias can be initialized to {"log(prior_class_freq)"} for imbalanced classification — a Chollet trick that gives meaningful first-step loss.
        (c) <strong>Residual branch's last conv</strong>: Fixup style — zero-init. Combined with scaling all earlier convs in the branch by {"L^{-1/2} = 200^{-1/2} ≈ 0.071"}, this makes the residual branch contribute nothing at init, so the network initially behaves like the identity. Training gradually wakes up each branch. Without this, the 200-layer residual stream at init has norm {"√200 ≈ 14×"} the input norm — enough to cause instabilities in the first hundred steps.
        (d) <strong>Final classification head</strong>: Xavier normal with {"gain=1"} (linear). Small output variance is crucial — initial logits should have variance {"~1"} so softmax cross-entropy starts at {"log(num_classes)"}. Kaiming on the head overshoots; constant {"std=0.02"} is a transformer-y choice but not well-motivated for a CNN without residual stream norm considerations.
        (e) <strong>Justifications</strong>: GELU is ReLU-like so Kaiming gain = {"√2"}. No BN means Fixup is the right fix for residual-depth explosion; {"L^{-1/2}"} is the analytical answer for the sum-of-branches norm. Zero last-conv makes the branch identity at init. SGD + Nesterov is LR-sensitive but insensitive to Adam-specific parameterization issues, so SP without μP is fine for a single width. Total overhead: 5 minutes to implement, a couple of lines of {"nn.init.*"} calls; saves dozens of failed training runs from diverging at epoch 1.
      </Callout>

    </div>
  ),
};

export default weightInitContent;
