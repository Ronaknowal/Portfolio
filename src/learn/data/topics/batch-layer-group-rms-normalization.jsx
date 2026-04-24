import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const normalizationContent = {
  title: "Batch/Layer/Group/RMS Normalization",
  readTime: "~40 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        For the first decade of modern deep learning, training a deep network was a mechanical exercise in superstition. You picked a learning rate that felt right, you initialized your weights with one of three recipes that were all folklore at the time, and if your twenty-layer convolutional net diverged on the third epoch you lowered the learning rate by half and tried again. This was not an exaggeration. The paper that invented batch normalization opens by describing how VGG-style networks were trained with carefully tuned learning rate schedules because any deviation from the schedule sent the activations of deep layers drifting into regions where gradients either exploded or died. The fix people used before 2015 was to train slowly, pray, and wait. The stated reason networks were hard to train was that the distribution of inputs to each layer kept shifting during training as the layers upstream of it updated their weights — a phenomenon the authors called internal covariate shift.
      </Prose>

      <Prose>
        Sergey Ioffe and Christian Szegedy published "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" at ICML 2015 (arXiv:1502.03167). Their proposal was surgical: take the activations of a layer, compute the mean and variance across the current mini-batch for each feature channel, subtract the mean, divide by the square root of the variance, and then — crucially — apply a learnable per-channel affine transform with parameters <Code>{"γ"}</Code> and <Code>{"β"}</Code>. The affine transform means the layer retains the capacity to undo the normalization entirely if that is what minimizes the loss, but in practice the optimization landscape becomes so much friendlier that it rarely wants to. The reported results were not subtle. Training time on ImageNet dropped by more than an order of magnitude. Dropout became optional. Higher learning rates became safe. BN was retrofitted into every serious architecture within months.
      </Prose>

      <Prose>
        The stated justification — internal covariate shift — turned out to be largely wrong. Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, and Aleksander Madry published "How Does Batch Normalization Help Optimization?" at NeurIPS 2018 (arXiv:1805.11604). Their central experiment was devastating in its simplicity. They trained networks with BN, and separately trained networks with BN followed by an injected layer of distributional noise designed to reintroduce as much covariate shift as the original non-BN baseline. If BN helps because it reduces internal covariate shift, the noise injection should destroy the benefit. It did not. The noised-BN network trained almost as well as the clean-BN one, and both crushed the no-BN baseline. The real mechanism is that BN smooths the loss landscape — it reduces the Lipschitz constant of the loss and of its gradient, which means gradient descent takes better steps. The field kept the technique and quietly dropped the original explanation.
      </Prose>

      <Prose>
        Once BN was accepted, the question became what to normalize over. The batch axis is a peculiar choice — it tangles every sample in a mini-batch together, creates a train-versus-eval discrepancy because no mini-batch is available at inference, and breaks as soon as the batch is small. Sequence models and recurrent networks were especially bad fits. Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton answered this in 2016 with "Layer Normalization" (arXiv:1607.06450). LayerNorm normalizes across the feature dimensions of a single sample. There is no batch entanglement, no running statistics, no train/eval mismatch. It works identically on a batch of one. It became the default in every architecture where samples have heterogeneous lengths or where batch is small — which, as of 2026, is essentially every Transformer ever trained.
      </Prose>

      <Prose>
        Two parallel lineages filled in the remaining corners of the design space. Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky published "Instance Normalization" in 2016 (arXiv:1607.08022) for neural style transfer, where the contents of each image should be normalized independently without any batch-level mixing. Yuxin Wu and Kaiming He published "Group Normalization" at ECCV 2018 (arXiv:1803.08494) for object detection and other settings where the per-GPU batch is too small for BN to be stable — GN splits the channels into groups and normalizes within each group per sample, giving most of the stability of LayerNorm with some of the channel-wise character of BN. Tim Salimans and Diederik Kingma contributed "Weight Normalization" in 2016 (arXiv:1602.07868) which reparameterizes the weight vector itself as a direction times a scalar magnitude; it is an interesting cousin but never reached the same adoption.
      </Prose>

      <Prose>
        The final turn came from the Transformer era. Biao Zhang and Rico Sennrich published "Root Mean Square Layer Normalization" (arXiv:1910.07467) at NeurIPS 2019. Their observation was that LayerNorm does two things — center the activations around zero and scale them to unit variance — and that only the scaling step matters for the downstream dynamics. Dropping the mean subtraction saves one reduction per layer, removes the learnable bias parameter <Code>{"β"}</Code>, and, they showed, loses essentially nothing in model quality. A second paper that year, Xiong et al.'s "On Layer Normalization in the Transformer Architecture" (arXiv:2002.04745), independently established that placing LayerNorm <em>before</em> each residual block (pre-norm) rather than after (post-norm) makes deep Transformers trainable without carefully tuned warm-up schedules. The combination — pre-norm placement with RMSNorm — is what every frontier LLM trained after 2022 uses. Llama, Gemma, Qwen, Mistral, and GPT-style decoders ship with it by default.
      </Prose>

      <Callout type="insight">
        Normalization is not a single trick. It is a family of answers to the question "what axis should we divide variance out along?" Each answer has a preferred regime. BN loves large batches of images. LN loves Transformers. GN loves detection models with per-GPU batches of two. RMSNorm loves LLMs that have to serve a billion tokens a second. Knowing which axis is right for the problem you are solving is closer to a literacy test than to a hyperparameter choice.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Every normalization layer does the same three operations in sequence. It computes a mean and a variance over some chosen axis of the activation tensor, it rescales the activations to have approximately zero mean and unit variance along that axis, and it applies a learnable per-feature affine transform so that the network can recover any scale and offset it actually wants. The only thing that changes between BatchNorm, LayerNorm, GroupNorm, InstanceNorm, and RMSNorm is the axis.
      </Prose>

      <Prose>
        Picture a convolutional activation tensor of shape <Code>(N, C, H, W)</Code> — batch, channels, height, width. There are four axes along which you could reduce. BatchNorm reduces over <Code>(N, H, W)</Code> for each channel — every channel gets its own scalar mean and variance, computed by averaging across all spatial locations in all samples in the batch. LayerNorm reduces over <Code>(C, H, W)</Code> for each sample — every sample gets its own scalar mean and variance, computed by averaging across all channels and spatial positions. InstanceNorm reduces over <Code>(H, W)</Code> per sample per channel — every sample-channel pair gets its own statistics. GroupNorm is the hybrid: split the channel axis into <Code>G</Code> groups of <Code>C/G</Code> channels each, then reduce over <Code>(C/G, H, W)</Code> within each group. GroupNorm recovers LayerNorm when <Code>G = 1</Code> and InstanceNorm when <Code>G = C</Code>.
      </Prose>

      <Heatmap
        label="normalization axes — which dims are reduced (shaded = reduced)"
        rowLabels={["BatchNorm", "LayerNorm", "InstanceNorm", "GroupNorm(G=4,C=8)", "RMSNorm"]}
        colLabels={["N (batch)", "C (channel)", "H (height)", "W (width)"]}
        matrix={[
          [1, 0, 1, 1],
          [0, 1, 1, 1],
          [0, 0, 1, 1],
          [0, 0.5, 1, 1],
          [0, 1, 1, 1],
        ]}
        colorScale="gold"
      />

      <Prose>
        For Transformers the picture is cleaner because there is no spatial dimension. Activations are shaped <Code>(B, T, D)</Code> — batch, tokens, model dim. LayerNorm and RMSNorm both reduce over the last axis <Code>D</Code>, giving each token its own scalar statistics. The difference is that LayerNorm subtracts the mean before dividing by the standard deviation; RMSNorm skips the mean subtraction and divides by the root-mean-square directly. In symbols, LN computes <Code>{"(x − μ)/σ"}</Code> and RMSNorm computes <Code>{"x / √(mean(x²) + ε)"}</Code>. If the mean of <Code>x</Code> is already near zero (which it tends to be for well-initialized Transformer residuals), the two quantities are nearly identical.
      </Prose>

      <Prose>
        Why does BN dominate in vision and LN in language? Two reasons, both practical. First, vision models of the BN era ran with batch sizes of 256 or 1024 per GPU, which gives BN plenty of samples to estimate per-channel statistics from. Language models run with sequence lengths in the thousands and batch sizes often in the single digits per GPU; LN does not care because it computes per-token. Second, BN's statistics are tangled across the batch in a way that makes causal language modeling awkward — the mean at position <Code>t</Code> would depend on future tokens in the batch, which is the kind of small mistake that takes a week of debugging to track down. LayerNorm has no such entanglement: each token's statistics come from its own feature vector only.
      </Prose>

      <Prose>
        The learnable affine at the end of every norm is easy to overlook but it is the piece that saves normalization from being lobotomizing. A network trained without that affine would be unable to produce activations with non-zero mean or non-unit variance even when the loss clearly wanted it to. With the affine, the network retains full expressive power — it can produce any scale and any offset it wants. The normalization step just puts the intermediate representation into a friendly numerical range before the network decides, one element-wise multiply and add later, what it actually wants to do with it.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 The shared template</H3>

      <Prose>
        Every one of BN, LN, GN, IN, and RMSNorm is a special case of the same scheme. Given a set of activation values <Code>{"S ⊂ ℝ"}</Code> (the set over which we reduce), compute the mean and variance of <Code>S</Code>, normalize every element of <Code>S</Code> to zero mean and unit variance (modulo a small <Code>ε</Code> for numerical safety), and apply a per-feature affine map.
      </Prose>

      <MathBlock>
        {"\\mu_S = \\tfrac{1}{|S|}\\sum_{x \\in S} x, \\qquad \\sigma_S^2 = \\tfrac{1}{|S|}\\sum_{x \\in S} (x - \\mu_S)^2"}
      </MathBlock>

      <MathBlock>
        {"\\hat{x} = \\frac{x - \\mu_S}{\\sqrt{\\sigma_S^2 + \\varepsilon}}, \\qquad y = \\gamma \\cdot \\hat{x} + \\beta"}
      </MathBlock>

      <Prose>
        The only thing that changes between norms is the definition of <Code>S</Code>. For a tensor indexed <Code>(n, c, h, w)</Code>:
      </Prose>

      <MathBlock>
        {"S_{\\text{BN}}(c) = \\{x_{n,c,h,w} : \\forall n, h, w\\}"}
      </MathBlock>

      <MathBlock>
        {"S_{\\text{LN}}(n) = \\{x_{n,c,h,w} : \\forall c, h, w\\}"}
      </MathBlock>

      <MathBlock>
        {"S_{\\text{IN}}(n, c) = \\{x_{n,c,h,w} : \\forall h, w\\}"}
      </MathBlock>

      <MathBlock>
        {"S_{\\text{GN}}(n, g) = \\{x_{n,c,h,w} : c \\in \\text{group}(g), \\forall h, w\\}"}
      </MathBlock>

      <Prose>
        The affine scale <Code>{"γ"}</Code> and shift <Code>{"β"}</Code> are always per-channel in BN/GN/IN and per-feature in LN. Their gradients are straightforward local products — nothing subtle happens there. The subtle thing is that the gradient of the loss with respect to <Code>x</Code> flows through <em>both</em> <Code>{"x̂"}</Code> and the statistics <Code>{"μ"}</Code> and <Code>{"σ²"}</Code> that depend on <Code>x</Code>. That cross-coupling is what gives normalization its optimization-smoothing property; it is also what makes the backward pass slightly nontrivial.
      </Prose>

      <H3>3.2 BN in train vs. eval: two different functions</H3>

      <Prose>
        BatchNorm is unusual among layers because its behavior at training time and evaluation time is literally different code paths. At training, BN computes per-batch statistics from the current mini-batch and uses them to normalize. While it does so, it also maintains an exponential moving average of the mean and variance over the course of training, controlled by a momentum parameter (PyTorch's default is <Code>0.1</Code>, which is the weight placed on the <em>new</em> value, not the old one — this tripped a generation of researchers).
      </Prose>

      <MathBlock>
        {"\\mu_{\\text{run}} \\leftarrow (1 - m)\\,\\mu_{\\text{run}} + m\\,\\mu_{\\text{batch}}"}
      </MathBlock>

      <MathBlock>
        {"\\sigma^2_{\\text{run}} \\leftarrow (1 - m)\\,\\sigma^2_{\\text{run}} + m\\,\\hat{\\sigma}^2_{\\text{batch}}"}
      </MathBlock>

      <Prose>
        where <Code>{"m"}</Code> is the momentum and <Code>{"\\hat{\\sigma}^2"}</Code> is the unbiased estimator used for inference. At evaluation time, BN stops computing per-batch statistics and uses the stored running mean and variance instead. That is the whole story — and also the source of most BN bugs. If the distribution of data at inference differs from what the running statistics were estimated on, or if someone forgets to call <Code>model.eval()</Code> and BN accidentally normalizes a test batch using only one sample's own statistics, results silently degrade. The failure mode is not a crash but a drift.
      </Prose>

      <Callout type="info" title="Momentum convention gotcha">
        PyTorch's <Code>{"momentum"}</Code> parameter is the weight on the <em>new</em> batch statistic, opposite to the convention used in some optimizer APIs. A momentum of <Code>0.1</Code> means the running stat moves 10% toward the current batch. A momentum of <Code>0</Code> means it never updates. This is the opposite of what the word suggests in most other contexts.
      </Callout>

      <H3>3.3 LayerNorm, RMSNorm, and why the mean barely matters</H3>

      <Prose>
        For a Transformer token with feature vector <Code>{"x \\in \\mathbb{R}^D"}</Code>, LayerNorm computes:
      </Prose>

      <MathBlock>
        {"\\text{LN}(x) = \\gamma \\odot \\frac{x - \\mu(x)}{\\sqrt{\\sigma^2(x) + \\varepsilon}} + \\beta"}
      </MathBlock>

      <Prose>
        RMSNorm computes:
      </Prose>

      <MathBlock>
        {"\\text{RMS}(x) = \\gamma \\odot \\frac{x}{\\sqrt{\\frac{1}{D}\\sum_i x_i^2 + \\varepsilon}}"}
      </MathBlock>

      <Prose>
        The RMSNorm formulation drops both the mean subtraction and the bias <Code>{"β"}</Code>. Zhang and Sennrich's empirical argument is that for modern Transformers the inputs to each norm have near-zero mean already — the residual connection plus the initialization conspire to keep running means close to zero — so the mean subtraction is a reduction that does almost nothing. The constant cost of that reduction (an allreduce across the feature dim, a subtraction, and the bookkeeping) is nontrivial at scale. Drop it and you save a small percentage of throughput per layer, multiplied by every layer, in every forward and backward pass. Over a training run the savings are real money.
      </Prose>

      <H3>3.4 Pre-norm vs. post-norm placement</H3>

      <Prose>
        The original Transformer ("Attention Is All You Need," Vaswani et al. 2017) placed LayerNorm <em>after</em> each sublayer, with the residual connection running around both:
      </Prose>

      <MathBlock>
        {"x_{\\ell+1} = \\text{LN}(x_\\ell + \\text{Sublayer}(x_\\ell)) \\quad \\text{(post-norm)}"}
      </MathBlock>

      <Prose>
        This works fine for twelve layers. It is treacherous for a hundred. Xiong et al. 2020 analyzed the gradient norms of the two placements and showed that post-norm's expected gradient at initialization grows with depth, while pre-norm's is roughly constant. Pre-norm flips the order:
      </Prose>

      <MathBlock>
        {"x_{\\ell+1} = x_\\ell + \\text{Sublayer}(\\text{LN}(x_\\ell)) \\quad \\text{(pre-norm)}"}
      </MathBlock>

      <Prose>
        With pre-norm, the residual stream is never normalized directly — gradient information flows through the identity connection unscaled, which makes deep networks stable without elaborate warm-up schedules. Baevski and Auli 2019 had reported this empirically for language modeling ("Adaptive Input Representations"); Xiong et al. made the theoretical case rigorous. Every modern LLM uses pre-norm. A final single LayerNorm is often applied after the last block to clean up the residual stream before the output head, because the residual stream under pre-norm can accumulate magnitude drift across blocks.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All the code in this section was run end-to-end against PyTorch's reference implementations. The numerical differences reported are the actual measured <Code>max|y_ours − y_ref|</Code> values. If you copy any of this into a notebook you should see identical results to within floating-point noise.
      </Prose>

      <H3>4.1 BatchNorm2d from scratch</H3>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn

torch.manual_seed(0)

def batch_norm_2d_scratch(x, gamma, beta, running_mean, running_var,
                          training=True, eps=1e-5, momentum=0.1):
    if training:
        # reduce over N, H, W per channel C
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        var  = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        n = x.numel() / x.size(1)                       # elements per channel
        running_mean.mul_(1 - momentum).add_(momentum * mean.squeeze())
        # PyTorch stores the unbiased variance in running_var
        running_var.mul_(1 - momentum).add_(momentum * var.squeeze() * n / (n - 1))
        x_hat = (x - mean) / torch.sqrt(var + eps)
    else:
        x_hat = ((x - running_mean.view(1, -1, 1, 1))
                 / torch.sqrt(running_var.view(1, -1, 1, 1) + eps))
    return gamma.view(1, -1, 1, 1) * x_hat + beta.view(1, -1, 1, 1)

C = 4
x = torch.randn(8, C, 5, 5)
bn_ref = nn.BatchNorm2d(C, eps=1e-5, momentum=0.1)
bn_ref.train()

gamma = bn_ref.weight.detach().clone()
beta  = bn_ref.bias.detach().clone()
rm    = bn_ref.running_mean.detach().clone()
rv    = bn_ref.running_var.detach().clone()

y_ref  = bn_ref(x)
y_ours = batch_norm_2d_scratch(x, gamma, beta, rm, rv, training=True)

print('=== BatchNorm2d ===')
print('max |y_ours - y_ref|:', (y_ours - y_ref).abs().max().item())
print('max |rm_ours - rm_ref|:', (rm - bn_ref.running_mean).abs().max().item())
print('max |rv_ours - rv_ref|:', (rv - bn_ref.running_var).abs().max().item())
# Output:
# === BatchNorm2d ===
# max |y_ours - y_ref|: 2.384185791015625e-07
# max |rm_ours - rm_ref|: 1.862645149230957e-09
# max |rv_ours - rv_ref|: 0.0`}
      </CodeBlock>

      <Prose>
        Two details are worth lingering on. First, the running variance update multiplies by <Code>{"n / (n - 1)"}</Code>. This is because the in-batch variance is computed with the biased (population) estimator — divide by <Code>n</Code>, not <Code>n-1</Code> — but the running variance stored for inference is the unbiased (sample) estimator because inference treats the training corpus as a sample from a larger distribution. Forgetting this correction is a common off-by-factor bug that produces correct training but subtly wrong evaluation. Second, the in-place updates to <Code>running_mean</Code> and <Code>running_var</Code> are intentional and match PyTorch's behavior — these buffers live outside the parameter set and are mutated by forward passes in training mode.
      </Prose>

      <H3>4.2 LayerNorm from scratch</H3>

      <CodeBlock language="python">
{`def layer_norm_scratch(x, gamma, beta, normalized_shape, eps=1e-5):
    # normalize over the last len(normalized_shape) dims
    dims = tuple(range(-len(normalized_shape), 0))
    mean = x.mean(dim=dims, keepdim=True)
    var  = x.var(dim=dims, keepdim=True, unbiased=False)
    x_hat = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_hat + beta

normalized_shape = (8,)
ln_ref = nn.LayerNorm(normalized_shape, eps=1e-5)
x = torch.randn(4, 6, 8)
y_ref  = ln_ref(x)
y_ours = layer_norm_scratch(x, ln_ref.weight, ln_ref.bias, normalized_shape)

print('=== LayerNorm ===')
print('max |y_ours - y_ref|:', (y_ours - y_ref).abs().max().item())
# Output:
# === LayerNorm ===
# max |y_ours - y_ref|: 2.384185791015625e-07`}
      </CodeBlock>

      <Prose>
        Note that LayerNorm's <Code>normalized_shape</Code> is a tuple specifying the trailing dimensions to normalize over. The most common case is <Code>(D,)</Code> — normalize each token's feature vector independently. You could pass <Code>(T, D)</Code> to normalize across tokens too, but almost nobody does; it would entangle unrelated tokens in a way that violates the per-token independence of the Transformer block.
      </Prose>

      <H3>4.3 GroupNorm from scratch</H3>

      <CodeBlock language="python">
{`def group_norm_scratch(x, gamma, beta, num_groups, eps=1e-5):
    N, C, H, W = x.shape
    G = num_groups
    # reshape so the channel axis splits into (G, C/G)
    x_g = x.view(N, G, C // G, H, W)
    mean = x_g.mean(dim=(2, 3, 4), keepdim=True)
    var  = x_g.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
    x_g  = (x_g - mean) / torch.sqrt(var + eps)
    x_hat = x_g.view(N, C, H, W)
    return gamma.view(1, C, 1, 1) * x_hat + beta.view(1, C, 1, 1)

C, G = 8, 4
gn_ref = nn.GroupNorm(G, C, eps=1e-5)
x = torch.randn(2, C, 6, 6)
y_ref  = gn_ref(x)
y_ours = group_norm_scratch(x, gn_ref.weight, gn_ref.bias, G)

print('=== GroupNorm ===')
print('num_groups =', G, ', num_channels =', C, ', per_group =', C // G)
print('max |y_ours - y_ref|:', (y_ours - y_ref).abs().max().item())
# Output:
# === GroupNorm ===
# num_groups = 4 , num_channels = 8 , per_group = 2
# max |y_ours - y_ref|: 2.384185791015625e-07`}
      </CodeBlock>

      <H3>4.4 RMSNorm from scratch</H3>

      <CodeBlock language="python">
{`def rms_norm_scratch(x, gamma, eps=1e-6):
    # normalize across the last dimension only
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    x_hat = x * torch.rsqrt(ms + eps)
    return gamma * x_hat

gamma = torch.ones(8)
x = torch.randn(2, 3, 8)

y_ours = rms_norm_scratch(x, gamma, eps=1e-6)

# Verify against PyTorch 2.4+ torch.nn.functional.rms_norm
y_torch = torch.nn.functional.rms_norm(x, normalized_shape=(8,),
                                       weight=gamma, eps=1e-6)

print('=== RMSNorm ===')
print('max |y_ours - torch.F.rms_norm|:', (y_ours - y_torch).abs().max().item())
# Output:
# === RMSNorm ===
# max |y_ours - torch.F.rms_norm|: 0.0`}
      </CodeBlock>

      <Prose>
        <Code>torch.rsqrt</Code> is the reciprocal square root — one operation instead of a sqrt followed by a division. On a GPU it maps to a single hardware instruction. Using it directly is marginally faster than writing <Code>{"1 / torch.sqrt(...)"}</Code>, and it is the form every production RMSNorm kernel ships with.
      </Prose>

      <H3>4.5 BN train/eval divergence with a single-sample batch</H3>

      <CodeBlock language="python">
{`bn = nn.BatchNorm2d(3, momentum=0.1)

# train for 5 steps on random data with non-zero mean
bn.train()
for _ in range(5):
    x = torch.randn(4, 3, 4, 4) + 2.0
    _ = bn(x)

print('After 5 training steps:')
print('running_mean:', bn.running_mean.detach().numpy().round(4))
print('running_var :', bn.running_var.detach().numpy().round(4))

x_single = torch.randn(1, 3, 4, 4) + 2.0
bn.eval()
y_eval = bn(x_single)
print('eval mode, output mean:', y_eval.mean().item())
print('eval mode, output std :', y_eval.std().item())

bn.train()
y_train = bn(x_single)
print('train mode (batch=1), output mean:', y_train.mean().item())
print('train mode (batch=1), output std :', y_train.std().item())
# Output:
# After 5 training steps:
# running_mean: [0.8288 0.8551 0.8164]
# running_var : [1.0523 1.0555 0.9996]
# eval mode, output mean: 1.0917738676071167
# eval mode, output std : 0.9334790706634521
# train mode (batch=1), output mean: 7.202228147207279e-08
# train mode (batch=1), output std : 1.0105763673782349`}
      </CodeBlock>

      <Prose>
        In training mode with a batch of one, the batch-axis variance collapses to zero (there is nothing to compute variance <em>over</em>), and every activation gets subtracted by itself and divided by <Code>{"\\sqrt{\\varepsilon}"}</Code>. The output mean is essentially numerical noise near zero. In evaluation mode, the same input is normalized against the running statistics accumulated during training, which gives a sensible distribution. This is the canonical "why is my model's test accuracy awful" bug and it is always caused by leaving BatchNorm in training mode at inference.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production</H2>

      <Prose>
        In practice you never hand-roll normalization in production. PyTorch ships battle-tested implementations with fused kernels; the only interesting decisions are which variant to instantiate and what to pass it.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn

# -------- BatchNorm2d: the vision default --------
bn = nn.BatchNorm2d(
    num_features=64,          # number of channels C
    momentum=0.1,             # weight on the NEW batch stat
    eps=1e-5,                 # for sqrt(var + eps) numerical safety
    affine=True,              # learn gamma and beta (defaults to True)
    track_running_stats=True, # maintain EMA for inference (defaults to True)
)

# -------- LayerNorm: the Transformer default --------
ln = nn.LayerNorm(
    normalized_shape=768,     # or a tuple to normalize over the last k dims
    eps=1e-5,
    elementwise_affine=True,  # learnable gamma and beta
)

# -------- GroupNorm: the small-batch-vision choice --------
gn = nn.GroupNorm(
    num_groups=32,            # must divide num_channels
    num_channels=64,
    eps=1e-5,
    affine=True,
)

# -------- RMSNorm: the modern LLM default --------
# PyTorch 2.4+ ships a native module:
rms = nn.RMSNorm(
    normalized_shape=4096,
    eps=1e-6,                 # LLMs typically use 1e-6, not 1e-5
    elementwise_affine=True,  # only gamma; there is no beta
)

# For older PyTorch, a manual implementation is a one-liner:
class RMSNormManual(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                * self.weight)`}
      </CodeBlock>

      <H3>5.1 Pre-norm Transformer block — the modern pattern</H3>

      <Prose>
        Every frontier LLM released since 2023 uses essentially the same block: pre-norm with RMSNorm, a rotary-positioned attention sublayer, a SwiGLU or GELU feed-forward sublayer, residual connections around both.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn

class PreNormBlock(nn.Module):
    """Modern pre-norm Transformer block. Llama-style when norm_cls=RMSNorm."""
    def __init__(self, d_model, n_heads, norm_cls=nn.RMSNorm, ffn_mult=4):
        super().__init__()
        self.norm_attn = norm_cls(d_model)
        self.attn      = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_ffn  = norm_cls(d_model)
        self.ffn       = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x):
        # note: norm BEFORE attention, residual AROUND the whole sublayer
        h = self.norm_attn(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a                           # residual #1
        x = x + self.ffn(self.norm_ffn(x))  # residual #2
        return x

block = PreNormBlock(d_model=512, n_heads=8, norm_cls=nn.RMSNorm)
x = torch.randn(4, 128, 512)
y = block(x)
print(y.shape)  # torch.Size([4, 128, 512])`}
      </CodeBlock>

      <H3>5.2 SyncBatchNorm for distributed training</H3>

      <Prose>
        If you must use BatchNorm in a multi-GPU training job, the per-GPU batch is usually too small for BN's per-device statistics to match the full-batch distribution. The fix is <Code>nn.SyncBatchNorm</Code>, which performs an <Code>allreduce</Code> across devices to compute the global mean and variance for every forward pass. This adds a collective communication per BN layer per step — typically the largest source of allreduce cost in a SyncBN training job. Convert an existing model in place:
      </Prose>

      <CodeBlock language="python">
{`# Convert every BatchNorm*d in a model to SyncBatchNorm
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)`}
      </CodeBlock>

      <Callout type="info" title="In practice">
        For distributed training with small per-GPU batch, most teams prefer GroupNorm or LayerNorm over SyncBatchNorm. The allreduce cost of SyncBN is nontrivial, and GN/LN have zero cross-device communication while delivering comparable accuracy on most vision tasks. SyncBN is still the right choice when you need to match a pretrained BN backbone exactly.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 Loss curves with and without normalization</H3>

      <Prose>
        The headline effect of normalization is that it makes training stable at much higher learning rates. Below is a schematic of what loss curves typically look like for a moderately deep CNN trained with and without BatchNorm at a matched learning rate. The numbers are illustrative but the shape is faithful: no-norm runs get stuck or diverge at aggressive learning rates while normalized runs sail through.
      </Prose>

      <Plot
        label="training loss: BN vs no-norm at matched learning rate"
        xLabel="epoch"
        yLabel="loss"
        series={[
          { name: "no norm", color: "#f87171", points: [[0, 2.30], [1, 2.27], [2, 2.25], [3, 2.24], [4, 2.24], [5, 2.23], [6, 2.23], [7, 2.22], [8, 2.22], [9, 2.21]] },
          { name: "BatchNorm", color: colors.gold, points: [[0, 2.30], [1, 1.85], [2, 1.40], [3, 1.05], [4, 0.80], [5, 0.65], [6, 0.55], [7, 0.48], [8, 0.44], [9, 0.41]] },
          { name: "LayerNorm", color: colors.green, points: [[0, 2.30], [1, 1.88], [2, 1.45], [3, 1.12], [4, 0.88], [5, 0.72], [6, 0.62], [7, 0.54], [8, 0.49], [9, 0.45]] },
        ]}
      />

      <H3>6.2 Heatmap: an activation tensor before and after BN</H3>

      <Prose>
        A small <Code>(N=4) × (C=6)</Code> slice of a convolutional feature map, treated here as a matrix of mean activations per sample-channel pair. Before BN, the channel means are all over the place. After BN, every channel's mean across the batch is zero and the variance is one — the affine transform is then free to scale and shift each channel back to wherever the loss prefers.
      </Prose>

      <Heatmap
        label="activation means — before BN (raw per-channel statistics)"
        rowLabels={["sample 0", "sample 1", "sample 2", "sample 3"]}
        colLabels={["ch 0", "ch 1", "ch 2", "ch 3", "ch 4", "ch 5"]}
        matrix={[
          [ 2.1, -0.4,  5.3,  1.2,  -2.5,  0.8],
          [ 1.7, -0.6,  4.8,  1.5,  -2.1,  0.5],
          [ 2.3, -0.2,  5.1,  0.9,  -2.7,  1.1],
          [ 1.9, -0.3,  5.0,  1.3,  -2.3,  0.7],
        ]}
        colorScale="warm"
      />

      <Heatmap
        label="activation means — after BN (zero per-channel mean, unit variance)"
        rowLabels={["sample 0", "sample 1", "sample 2", "sample 3"]}
        colLabels={["ch 0", "ch 1", "ch 2", "ch 3", "ch 4", "ch 5"]}
        matrix={[
          [ 0.45, -0.32,  1.12, -0.68,  -0.45,  0.42],
          [-1.35, -1.59, -0.87,  0.90,   1.35, -1.26],
          [ 1.35,  1.59,  0.22, -1.58,  -1.35,  1.68],
          [-0.45,  0.32, -0.47,  1.36,   0.45, -0.84],
        ]}
        colorScale="green"
      />

      <H3>6.3 StepTrace: one BatchNorm forward pass</H3>

      <StepTrace
        label="BN forward pass with running stats update"
        steps={[
          {
            label: "Input batch",
            render: () => (
              <div>
                <Prose>
                  Input tensor of shape <Code>(N=4, C=3, H=2, W=2)</Code>. Each channel has <Code>N·H·W = 16</Code> activations that will be reduced to a single mean and variance.
                </Prose>
                <CodeBlock language="python">{`x.shape == (4, 3, 2, 2)
# 16 activation values per channel`}</CodeBlock>
              </div>
            ),
          },
          {
            label: "Per-channel statistics",
            render: () => (
              <div>
                <Prose>
                  Reduce over batch and spatial axes. Each channel gets a scalar mean and variance.
                </Prose>
                <CodeBlock language="python">{`mean = x.mean(dim=(0, 2, 3), keepdim=True)  # shape (1, 3, 1, 1)
var  = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)  # (1, 3, 1, 1)`}</CodeBlock>
              </div>
            ),
          },
          {
            label: "Normalize",
            render: () => (
              <div>
                <Prose>
                  Subtract the per-channel mean, divide by the per-channel standard deviation (with epsilon).
                </Prose>
                <CodeBlock language="python">{`x_hat = (x - mean) / torch.sqrt(var + 1e-5)
# now every channel has mean ~= 0, std ~= 1`}</CodeBlock>
              </div>
            ),
          },
          {
            label: "Affine rescale",
            render: () => (
              <div>
                <Prose>
                  Apply the learnable per-channel scale and shift.
                </Prose>
                <CodeBlock language="python">{`y = gamma.view(1, -1, 1, 1) * x_hat + beta.view(1, -1, 1, 1)`}</CodeBlock>
              </div>
            ),
          },
          {
            label: "Update running statistics",
            render: () => (
              <div>
                <Prose>
                  In training mode, blend the current batch stats into the stored running statistics with momentum <Code>m = 0.1</Code>. The unbiased correction <Code>n/(n−1)</Code> is applied to variance so that inference uses the sample estimator.
                </Prose>
                <CodeBlock language="python">{`running_mean = (1 - 0.1) * running_mean + 0.1 * mean.squeeze()
running_var  = (1 - 0.1) * running_var  + 0.1 * var.squeeze() * n / (n - 1)`}</CodeBlock>
              </div>
            ),
          },
          {
            label: "Next forward pass (eval)",
            render: () => (
              <div>
                <Prose>
                  At evaluation time, BN stops computing batch statistics and uses the stored running statistics directly. No running-stats update, no train/eval mismatch — provided <Code>model.eval()</Code> was called.
                </Prose>
                <CodeBlock language="python">{`x_hat = ((x - running_mean.view(1, -1, 1, 1))
         / torch.sqrt(running_var.view(1, -1, 1, 1) + 1e-5))
y = gamma.view(1, -1, 1, 1) * x_hat + beta.view(1, -1, 1, 1)`}</CodeBlock>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Choosing a normalization is usually determined by two properties of the problem: the architecture family (vision vs. sequence vs. generative) and the effective per-device batch size. The table below captures the modern defaults.
      </Prose>

      <Heatmap
        label="normalization by scenario (darker = stronger preference)"
        rowLabels={[
          "CNN, batch ≥ 32",
          "CNN, batch < 8 (detection)",
          "Transformer / LLM",
          "RNN / LSTM",
          "Style transfer / GAN generator",
          "Distributed, small per-GPU batch",
          "Diffusion UNet",
        ]}
        colLabels={["BatchNorm", "LayerNorm", "GroupNorm", "InstanceNorm", "RMSNorm", "SyncBN"]}
        matrix={[
          [1.0, 0.2, 0.4, 0.1, 0.1, 0.3],
          [0.2, 0.5, 1.0, 0.3, 0.1, 0.4],
          [0.0, 0.7, 0.1, 0.0, 1.0, 0.0],
          [0.1, 1.0, 0.2, 0.0, 0.3, 0.0],
          [0.2, 0.3, 0.5, 1.0, 0.1, 0.0],
          [0.2, 0.6, 1.0, 0.2, 0.3, 0.7],
          [0.1, 0.3, 1.0, 0.2, 0.1, 0.0],
        ]}
        colorScale="gold"
      />

      <H3>7.1 Rules of thumb</H3>

      <Prose>
        The CNN defaults still belong to BatchNorm when you have the batch for it. ResNets, EfficientNets, and the great majority of ImageNet-scale classifiers ship with BN because at batch sizes of 256 or higher the per-channel statistics are tight and BN's slight regularization effect is a free benefit. As soon as you drop into detection or segmentation, where per-GPU batches of 2–4 are standard because of large input resolution, switch to GroupNorm with <Code>G = 32</Code>. This is the Wu and He recommendation and it is battle-tested on Mask R-CNN, RetinaNet, and every modern detector.
      </Prose>

      <Prose>
        Transformers pick between LayerNorm and RMSNorm. For a new architecture from scratch in 2026, RMSNorm is the default — it matches LN's quality, saves a small amount of throughput, and is what every serious open LLM (Llama, Qwen, Gemma, Mistral) has standardized on. Stay with LayerNorm only if you are building on top of a pretrained encoder (BERT, T5, older RoBERTa) whose weights were trained with LN and whose checkpoints you want to load.
      </Prose>

      <Prose>
        RNNs and LSTMs take LayerNorm almost universally. Generative style-transfer networks use InstanceNorm because the whole point is to normalize per-image style independently of the batch. Diffusion UNets have converged on GroupNorm in the image-space backbone and LayerNorm in the Transformer blocks used for cross-attention. When distributed training gives you a small per-GPU batch and you are modifying a pretrained BN backbone, SyncBatchNorm is the right answer for fidelity — but plain GroupNorm is usually the better engineering choice because it avoids the allreduce cost entirely.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <Prose>
        Normalization has cost, and different variants have different scaling behavior as models and batches grow.
      </Prose>

      <H3>8.1 BN breaks at small batch</H3>

      <Prose>
        BatchNorm's per-channel variance estimate requires enough samples in the batch that the sample variance is a reasonable proxy for the true variance. At a per-GPU batch of 32 this is fine. At 8 it starts to wobble. At 2 — the typical per-GPU batch for object detection or high-resolution segmentation — the variance estimate is so noisy that BN becomes a liability rather than an asset. Wu and He's 2018 paper includes the key chart: BN's ImageNet top-1 degrades from ~76% at batch 32 to below 70% at batch 2, while GroupNorm stays flat around 75% across the entire range. This is why detection and segmentation standardized on GN.
      </Prose>

      <H3>8.2 SyncBatchNorm pays a communication cost</H3>

      <Prose>
        If you need BN-style statistics in a distributed setting, you call <Code>SyncBatchNorm.convert_sync_batchnorm</Code> and every BN layer now performs an allreduce across all GPUs on every forward pass to compute the global mean and variance. On a 1024-GPU training run this allreduce is latency-dominated — you are paying for the slowest GPU in the group plus network round-trip. Depending on topology, the collective cost of SyncBN layers can reach 5–15% of total step time for training-heavy workloads. This is the single biggest reason most new architectures avoid BN entirely.
      </Prose>

      <H3>8.3 RMSNorm saves real throughput at LLM scale</H3>

      <Prose>
        RMSNorm skips one reduction (the mean) and removes one parameter vector (the bias <Code>β</Code>). Each saved reduction on a large LLM feature vector represents a few microseconds per layer per step. Across 96 layers and hundreds of millions of forward passes over a multi-trillion-token training run, the savings compound into real hours of wall-clock time and real dollars of hardware. Zhang and Sennrich report 7–64% per-layer speedups for the norm operation itself depending on the setting; the end-to-end training speedup is typically 10–15% on top of a LayerNorm baseline.
      </Prose>

      <H3>8.4 Fused norms are how production gets the rest of the way there</H3>

      <Prose>
        A naive LayerNorm implementation does several sequential operations — a mean reduction, a subtraction, a squared-mean reduction, a rsqrt, a multiply, and an affine apply. Every one of these is a separate kernel launch and a separate memory traversal of the activation tensor. For <Code>(B, T, D)</Code> = <Code>(8, 8192, 8192)</Code>, just reading the tensor once is 2 GB of memory traffic. Fused norm kernels (NVIDIA Apex's <Code>FusedLayerNorm</Code>, the FlashNorm family from the FlashAttention authors, PyTorch's own fused implementations in newer releases) compute all of these steps in a single kernel with on-chip accumulation — one read, one write. The speedup over the naive implementation is typically 3–5x for the norm operation itself, and since norms are called a dozen times per Transformer layer, this is not cosmetic. Every serious training stack uses fused norms by default.
      </Prose>

      <H3>8.5 GroupNorm scales with groups, not channels</H3>

      <Prose>
        GroupNorm's compute cost is the same as LayerNorm for a fixed activation tensor — one reduction per group per sample. Its memory and compute both scale linearly with the number of elements, which means GN is asymptotically free compared to the convolutions it sits between. The only scaling consideration is that <Code>num_groups</Code> must divide <Code>num_channels</Code>. The standard choice is <Code>G = 32</Code>, which gives each group enough channels (typically 4–32) for the group statistics to be stable. Very small or very large group counts are both worse.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 BN at inference with a different batch distribution</H3>

      <Prose>
        BN's running statistics are estimated on the training distribution. If you deploy your model on a stream of images that differ systematically from training — different lighting, different camera, different preprocessing — the stored running mean and variance are wrong for this data, and BN will silently shift the activations into the wrong regime. Every downstream layer inherits the mistake. The symptom is that evaluation accuracy is close to training accuracy on held-out training-distribution data but much worse on real deployment data. The fix is usually to re-estimate BN statistics on a small sample from the deployment distribution — either by running a few forward passes in training mode (dangerous, updates weights if gradients are enabled), or by a dedicated calibration pass that updates running stats only. PyTorch's <Code>{"torch.utils.data.DataLoader"}</Code> with <Code>model.train()</Code> and <Code>{"torch.no_grad()"}</Code> accomplishes this cleanly.
      </Prose>

      <H3>9.2 BN interacts badly with aggressive data augmentation</H3>

      <Prose>
        Strong augmentation — heavy cutout, mixup, random erasing — creates a training distribution that looks almost nothing like the test distribution at any given moment. BN's per-batch statistics during training are computed on this augmented distribution; the running statistics accumulated across training steps are a noisy average of many augmented distributions; the test distribution is clean. A well-tuned augmentation pipeline can therefore systematically bias BN's running stats. Most modern augmentation-heavy training recipes compensate by running a final few epochs with a low learning rate and reduced augmentation, which lets BN's stats settle on something closer to the clean distribution. Some recipes go further and re-estimate running stats from scratch over the clean training set after training ends.
      </Prose>

      <H3>9.3 BN in RNNs is a disaster</H3>

      <Prose>
        Naively applying BN to a recurrent network means computing statistics per time step per channel across the batch. Sequences are of different lengths, the distribution at early time steps is systematically different from the distribution at late time steps, and the whole mess is incompatible with variable-length padded sequences. The original BN paper explicitly avoided RNNs. This is exactly the niche LayerNorm was invented for — Ba, Kiros, and Hinton's original motivation was getting normalization into RNNs where BN could not go. Any new RNN-adjacent architecture should use LayerNorm or avoid normalization entirely.
      </Prose>

      <H3>9.4 GroupNorm with a group count that does not divide channels</H3>

      <CodeBlock language="python">
{`gn = nn.GroupNorm(3, 8)  # 3 does not divide 8
# -> ValueError: num_channels must be divisible by num_groups`}
      </CodeBlock>

      <Prose>
        GroupNorm requires <Code>num_channels % num_groups == 0</Code>. The standard choice is <Code>G = 32</Code>, which divides every power-of-two channel count from 32 upward. When transplanting GN into a model with non-standard channel widths, pick the largest <Code>G</Code> that divides every channel count in the network, or fall back to <Code>G = 1</Code> (which makes GN equivalent to LayerNorm across the channel dim).
      </Prose>

      <H3>9.5 Post-norm in deep Transformers</H3>

      <Prose>
        Post-norm Transformers — the original 2017 placement — do not train past roughly twelve layers without a carefully tuned learning rate warmup. The gradient norm at initialization grows with depth in post-norm, which means the optimizer needs to start with a tiny learning rate and ramp it up over thousands of steps. Pre-norm eliminates this: gradient norms are depth-independent at initialization, warmup becomes optional, and you can stack a hundred layers without special care. If you find yourself inheriting a post-norm architecture and wondering why training diverges above twenty layers, this is almost certainly the reason. Flipping to pre-norm is a three-line change and usually resolves it.
      </Prose>

      <H3>9.6 RMSNorm without the learnable scale</H3>

      <Prose>
        RMSNorm's whole quality story rests on the learnable per-feature scale <Code>γ</Code>. Stripped of <Code>γ</Code>, RMSNorm is just "divide by the feature-vector magnitude" — a hard constraint that the network has no way to undo. Models trained without the learnable scale are substantially worse than models trained with it; the difference is more than a percentage point of perplexity at LLM scale. The parameter is cheap (one vector per layer) and should always be enabled.
      </Prose>

      <H3>9.7 Mixing training-mode BN with gradient accumulation</H3>

      <Prose>
        Gradient accumulation — multiple small forward-backward passes whose gradients are summed before a single optimizer step — does not give BN a bigger effective batch. Each micro-batch produces its own BN statistics, and the running mean/variance update happens once per micro-batch. If you accumulate gradients over 8 micro-batches of size 4, BN sees a sequence of 8 independent batches of 4, not one batch of 32. This is a frequent source of confusion when trying to compensate for memory-limited GPUs by trading batch for accumulation. The only way to get "BN on a large effective batch" across devices is SyncBatchNorm; across accumulation steps on a single device, there is no such thing.
      </Prose>

      <H3>9.8 Epsilon too small at low precision</H3>

      <Prose>
        All normalization formulas include a <Code>{"+ ε"}</Code> inside the square root for numerical safety. The default <Code>{"ε = 1e-5"}</Code> is fine for FP32 but is below the representable resolution of BF16 and risks denormal-flush at FP16. For low-precision training, <Code>{"ε = 1e-6"}</Code> or the layer-specific fused-norm kernel's internal epsilon (typically set in FP32 even when activations are BF16) is the safer choice. RMSNorm at scale uses <Code>{"ε = 1e-6"}</Code> by convention.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The literature here is unusually clean — each normalization has one canonical paper that introduces it, and one or two follow-ups that reinterpret or refine it. Read in this order if you want the full story.
      </Prose>

      <Prose>
        <strong>Ioffe and Szegedy (2015).</strong> "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." ICML 2015. arXiv:1502.03167. The paper that started everything. The stated mechanism (reducing internal covariate shift) is now understood to be largely wrong, but the technique and the running-statistics machinery are as described.
      </Prose>

      <Prose>
        <strong>Ba, Kiros, Hinton (2016).</strong> "Layer Normalization." arXiv:1607.06450. Introduced LayerNorm specifically to handle the case where BN's batch-axis dependence is problematic — RNNs, variable-length sequences, tiny batches.
      </Prose>

      <Prose>
        <strong>Ulyanov, Vedaldi, Lempitsky (2016).</strong> "Instance Normalization: The Missing Ingredient for Fast Stylization." arXiv:1607.08022. InstanceNorm for style transfer; short paper with a clean empirical demonstration.
      </Prose>

      <Prose>
        <strong>Wu and He (2018).</strong> "Group Normalization." ECCV 2018. arXiv:1803.08494. GroupNorm, motivated by the collapse of BN at small batches in detection and segmentation. Includes the definitive batch-size study showing GN's flatness versus BN's degradation.
      </Prose>

      <Prose>
        <strong>Zhang and Sennrich (2019).</strong> "Root Mean Square Layer Normalization." NeurIPS 2019. arXiv:1910.07467. RMSNorm; the paper argues that LN's mean subtraction is the recentering part and the rescaling part is what matters, and drops the former. Now the default in frontier LLMs.
      </Prose>

      <Prose>
        <strong>Santurkar, Tsipras, Ilyas, Madry (2018).</strong> "How Does Batch Normalization Help Optimization?" NeurIPS 2018. arXiv:1805.11604. The paper that overturned the internal-covariate-shift story by showing that deliberately reintroduced covariate shift does not destroy BN's benefit. Establishes loss-landscape smoothing as the actual mechanism.
      </Prose>

      <Prose>
        <strong>Xiong et al. (2020).</strong> "On Layer Normalization in the Transformer Architecture." ICML 2020. arXiv:2002.04745. Formal analysis of pre-norm versus post-norm placement. Explains why post-norm Transformers need warmup and pre-norm ones do not.
      </Prose>

      <Prose>
        <strong>Salimans and Kingma (2016).</strong> "Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks." NeurIPS 2016. arXiv:1602.07868. An alternative approach that reparameterizes weight vectors rather than activations. Historically interesting and still occasionally useful, though it has been eclipsed by activation norms in practice.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check</H2>

      <Prose>
        <strong>Question 1.</strong> You train a CNN with BatchNorm2d and a momentum of 0.1. After 1000 steps you stop training and evaluate. Your test accuracy is catastrophic, but your training accuracy was fine. What is the most likely one-line bug?
      </Prose>

      <Callout type="answer" title="Answer 1">
        You forgot to call <Code>model.eval()</Code>. In training mode BN uses per-batch statistics; at inference with a small or non-representative evaluation batch, those statistics are wildly different from what the downstream weights were trained against. Calling <Code>model.eval()</Code> switches BN to use the stored running statistics, which is what the rest of the network was trained to expect.
      </Callout>

      <Prose>
        <strong>Question 2.</strong> You are training a 48-layer Transformer with post-norm placement. The loss diverges to NaN on step 3 regardless of how small you make the learning rate. Without changing the learning rate schedule or the architecture, what one change would most likely fix it?
      </Prose>

      <Callout type="answer" title="Answer 2">
        Flip to pre-norm: move the LayerNorm inside each residual block so it precedes the sublayer rather than following the residual sum. Pre-norm Transformers have depth-independent gradient norms at initialization and train stably to hundreds of layers without special warmup, while post-norm is only stable to roughly a dozen layers (Xiong et al. 2020). The fix is literally <Code>{"x = x + sublayer(norm(x))"}</Code> instead of <Code>{"x = norm(x + sublayer(x))"}</Code>.
      </Callout>

      <Prose>
        <strong>Question 3.</strong> You are building a new LLM from scratch in 2026 and want maximum training throughput. You have a choice of LayerNorm or RMSNorm. Why would you pick RMSNorm, and what do you lose?
      </Prose>

      <Callout type="answer" title="Answer 3">
        RMSNorm drops the mean subtraction and the learnable bias <Code>β</Code>. The mean reduction is one fewer pass over the feature vector per forward/backward, which saves roughly 7–15% of the norm layer's wall-clock time and scales to a meaningful full-training speedup. The bias removal saves one parameter vector per norm (trivial memory but cleaner state management). You lose essentially nothing in model quality — Zhang and Sennrich (2019) and a decade of follow-up LLM training have shown the mean subtraction does not help Transformer quality in practice, because residual-stream means are already close to zero.
      </Callout>

      <Prose>
        <strong>Question 4.</strong> You inherit a detection model with BatchNorm2d layers that was trained on 8 GPUs with batch 2 per GPU. Training accuracy is decent but validation accuracy is terrible. You suspect BN. What are your two realistic fixes, and which is usually preferred?
      </Prose>

      <Callout type="answer" title="Answer 4">
        Fix A: replace every BN layer with SyncBatchNorm via <Code>{"nn.SyncBatchNorm.convert_sync_batchnorm(model)"}</Code>. This gives BN a global effective batch of 16 across the group, which is enough for stable statistics. Fix B: replace BN with GroupNorm (<Code>G = 32</Code>). GroupNorm has no batch-axis dependence at all, no cross-GPU communication, and matches or beats BN at small per-GPU batch (Wu and He 2018). GN is usually preferred because it is a pure per-sample operation with zero allreduce cost, while SyncBN pays an allreduce per BN layer per step.
      </Callout>

      <Prose>
        <strong>Question 5.</strong> You read a blog post claiming that normalization layers work because they "reduce internal covariate shift." A junior colleague asks if that is true. Give a precise one-paragraph answer.
      </Prose>

      <Callout type="answer" title="Answer 5">
        The internal-covariate-shift explanation is the one the original 2015 paper offered, and it has been largely overturned. Santurkar, Tsipras, Ilyas, and Madry (NeurIPS 2018) ran the decisive experiment: they injected distributional noise after every BN layer so that the resulting activations had <em>more</em> covariate shift than an un-normalized baseline, and the noised-BN network trained essentially as well as the clean-BN one. If BN's benefit came from reducing covariate shift, the noise should have destroyed it. The modern consensus is that BN helps because it smooths the loss landscape — it reduces the effective Lipschitz constant of the loss and of its gradient — which means gradient descent takes better steps at higher learning rates. The community kept the technique and quietly dropped the original explanation.
      </Callout>

    </div>
  ),
};

export default normalizationContent;
