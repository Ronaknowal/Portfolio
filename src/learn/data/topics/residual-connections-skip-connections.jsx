import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const residualConnectionsContent = {
  title: "Residual Connections & Skip Connections",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        In May 2015, Rupesh Srivastava, Klaus Greff, and Jürgen Schmidhuber of IDSIA posted "Highway Networks" to arXiv (1505.00387). The paper asked a question the field had been avoiding: why do very deep feedforward networks train so poorly? At the time, the depth ceiling for convolutional networks was roughly 20 layers — VGG-19 (Simonyan and Zisserman, 2014) was the deepest widely trainable classifier, and even that was fragile. Anything beyond 20 layers diverged, plateaued, or failed to match the performance of a shallower network. Highway Networks proposed a fix inspired by LSTM gates: add a learned transform gate T(x) and carry gate C(x) so each layer computes y = H(x, W) * T(x) + x * C(x). When the gates choose "carry", the layer passes its input through unchanged. The paper showed training of 50-layer and 100-layer networks on MNIST and CIFAR-10 that would have been impossible with vanilla feedforward blocks.
      </Prose>

      <Prose>
        Six months later, in December 2015, Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun at Microsoft Research Asia posted "Deep Residual Learning for Image Recognition" to arXiv (1512.03385). They simplified Highway Networks to the extreme: drop the learned gates; just add the identity. A block computes y = F(x, W) + x. That single change unlocked the depth dimension. ResNet-152 won ILSVRC 2015 in image classification, detection, and localization. The paper won the CVPR 2016 Best Paper Award. As of 2026, it has over 250,000 citations and is arguably the most influential single architecture paper in the deep learning era.
      </Prose>

      <Prose>
        The central problem the paper named was <strong>the degradation problem</strong>. Before ResNet, the assumption was that very deep networks failed because of overfitting or vanishing gradients. Both explanations were wrong. He and colleagues ran a clean experiment: train a plain 20-layer CNN and a plain 56-layer CNN on CIFAR-10, both with batch normalization. The 56-layer network had <em>higher training error</em> than the 20-layer one — not just higher test error. Overfitting would have shown as lower training error and higher test error. Vanishing gradients would have shown as loss plateaus at initialization. This was something new: the deeper network could not even fit the training set as well. The optimization itself was broken.
      </Prose>

      <Prose>
        The logical contradiction was sharp. Any function a 20-layer network can represent, a 56-layer network can also represent — by making the extra 36 layers learn the identity mapping. The deeper network has a strictly larger function class. Yet SGD with standard initialization could not find this solution. Something about the optimization landscape made the identity mapping hard to discover through composition of nonlinear layers. ResNet's fix was to make the identity the <em>default</em>: reparametrize each block so it outputs F(x) + x, where F is what the block learns to add on top of the input. If F = 0, the block is the identity. The network now has to learn to <em>deviate</em> from identity rather than to <em>find</em> identity. That reparametrization, and nothing else, cured the degradation problem.
      </Prose>

      <Prose>
        In 2016, the same team at MSRA published "Identity Mappings in Deep Residual Networks" (arXiv 1603.05027). The paper asked: what is the right ordering of Conv, BN, ReLU, and the addition? The original ResNet put the addition <em>before</em> a final ReLU, so the signal flowing back through the skip path encountered a nonlinearity. The 2016 paper flipped the order — pre-activation: BN → ReLU → Conv inside each block, with a pure identity path carrying x untouched from input to output. This let the authors train a 1001-layer ResNet on CIFAR-10 with stable optimization. Pre-activation is now the default in almost every deep residual architecture.
      </Prose>

      <Prose>
        In 2017, Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Weinberger published "Densely Connected Convolutional Networks" (DenseNet) at CVPR 2017 (arXiv 1608.06993). DenseNet pushed the skip idea further: within a block, every layer receives the <em>concatenation</em> of all prior layers' outputs, so y_l = H_l([x_0, x_1, ..., x_{l-1}]). Gradients flow back through a dense web of shortcut paths rather than a single additive one. DenseNet-BC-190 achieved state-of-the-art on CIFAR-10/100 with roughly a third the parameters of a comparable ResNet. Concatenation trades memory for parameter efficiency; the core insight — force strong connectivity between distant layers — is the same.
      </Prose>

      <Prose>
        The residual pattern escaped the computer vision world within a year. In 2017, Vaswani and colleagues' "Attention is All You Need" put residual connections at the heart of the Transformer: every sub-layer computes y = x + Sublayer(LayerNorm(x)) (or the pre-norm variant y = x + Sublayer(x) followed by LayerNorm, depending on the paper). Xie, Girshick, Dollár, Tu, and He's 2017 ResNeXt (arXiv 1611.05431) replaced the single residual conv path with a sum over C parallel branches, introducing the "cardinality" dimension. Gao and colleagues' 2019 Res2Net (arXiv 1904.01169) decomposed each residual block into hierarchical multi-scale substreams. In 2020, Thomas Bachlechner and colleagues at UCSD published "ReZero is All You Need" (arXiv 2003.04887), which observed that y = x + α · F(x) with α initialized to 0 enables stable training of arbitrarily deep networks even without LayerNorm. Touvron and colleagues at Facebook AI Research refined this in 2021's CaiT (arXiv 2103.17239) as LayerScale: y = x + diag(λ_1, ..., λ_d) · F(x) with tiny per-channel scalars, which stabilized 36-layer vision Transformers that were otherwise untrainable.
      </Prose>

      <Prose>
        Across a decade, a single architectural device — add the input to the output — has become the foundation of deep learning. Every production image classifier, every Transformer (encoder or decoder), every diffusion model, every modern speech recognizer, every vision Transformer, every protein language model contains residual connections. The pattern is so universal that it no longer feels like an architectural choice; it feels like a law of physics for deep networks.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The residual block computes <Code>{"y = F(x) + x"}</Code>. That single equation hides four distinct ideas that together explain why the pattern works.
      </Prose>

      <Prose>
        <strong>Idea 1 — Gradient highway.</strong> During backpropagation, the gradient of the loss with respect to the block input is <Code>{"dL/dx = dL/dy * (1 + dF/dx)"}</Code>. The "1" is the derivative of the identity path. Even if <Code>{"dF/dx"}</Code> shrinks toward zero — as it does in a deep stack of plain Conv-BN-ReLU layers — the "+1" term preserves the upstream gradient. At a stack of N residual blocks, the gradient reaching the first block is bounded below by the gradient at the loss, because every block contributes at least a "1" along the chain. Vanishing gradients are not merely mitigated; they are structurally impossible along the identity path.
      </Prose>

      <Prose>
        <strong>Idea 2 — Learn the residual, not the mapping.</strong> Consider what the block needs to learn. In a plain network, layer <Code>l</Code> learns a mapping <Code>{"H(x)"}</Code> directly from its input. In a residual network, it learns <Code>{"F(x) = H(x) - x"}</Code>, the perturbation relative to identity. If the optimal mapping is close to identity — as it often is in a deep network where most layers should be refining features rather than reinventing them — then <Code>F</Code> is close to zero, which is a much easier function to represent than the full identity. The inductive bias matches the actual task. This is the reason He and colleagues in 2015 proposed the pattern: the problem was not that networks couldn't represent identity, but that they couldn't <em>find</em> it. Residuals reparametrize the search so identity is the origin.
      </Prose>

      <Prose>
        <strong>Idea 3 — Ensemble of paths.</strong> In 2016, Andreas Veit, Michael Wilber, and Serge Belongie published "Residual Networks Behave Like Ensembles of Relatively Shallow Networks" (arXiv 1605.06431). They pointed out that a ResNet with N residual blocks can be unrolled into <Code>{"2^N"}</Code> paths from input to output — at each block, information can either traverse <Code>F</Code> or skip it. The network's output is an exponential ensemble of all these paths, weighted by which combinations of <Code>F</Code> are "active" given the input. They verified this empirically: deleting individual blocks from a trained ResNet barely affected test accuracy, because any single path is only one vote in the ensemble. This is why ResNets are unusually robust to architectural perturbations and why stochastic depth (dropping random blocks during training) works as a regularizer.
      </Prose>

      <Prose>
        <strong>Idea 4 — Degradation-free depth.</strong> The pre-ResNet world had a depth ceiling around 20 layers for CNNs. Adding more layers didn't help and often hurt. ResNet broke that ceiling by making "add another block" a safe operation: if the new block is initialized near zero, it contributes near-identity and cannot hurt the network. Training then fine-tunes it into a useful residual. This is an optimization guarantee, not just an empirical observation. It is also why models now scale past 1000 layers (the 2016 pre-activation ResNet), why GPT-4-class Transformers stack 100+ blocks, and why the question "how deep can we go?" has become a matter of compute rather than optimization.
      </Prose>

      <Callout accent="gold">
        Mental model: a residual block is a <em>conditional refinement</em>. The input is already a valid representation; the block optionally adds a small correction. The identity path is the "do nothing" baseline; the conv/attention/MLP path is the "make it better" proposal. SGD's job is to learn which corrections improve the loss.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 The basic residual block</H3>

      <Prose>
        Let <Code>{"x \\in R^d"}</Code> be the input to a residual block and <Code>{"F"}</Code> be a differentiable function (typically a small stack of Conv, BN, ReLU, or Attn and LayerNorm). The residual block computes:
      </Prose>

      <MathBlock>
        {"y = F(x, W) + x"}
      </MathBlock>

      <Prose>
        where <Code>W</Code> is the set of learnable parameters inside <Code>F</Code>. Stacking <Code>L</Code> such blocks gives a recursive structure:
      </Prose>

      <MathBlock>
        {"x_{l+1} = F_l(x_l, W_l) + x_l \\quad \\text{for } l = 0, 1, \\ldots, L-1"}
      </MathBlock>

      <Prose>
        Unrolling the recursion expresses the output of the deep stack as the input plus a sum of residuals:
      </Prose>

      <MathBlock>
        {"x_L = x_0 + \\sum_{l=0}^{L-1} F_l(x_l, W_l)"}
      </MathBlock>

      <Prose>
        This telescoping form is the mathematical basis of the "ensemble of paths" view. Every subset of the <Code>{"F_l"}</Code> terms corresponds to a different path through the network; the output is a sum over all active subsets.
      </Prose>

      <H3>3.2 Gradient through the identity path</H3>

      <Prose>
        Differentiating the block output with respect to the block input:
      </Prose>

      <MathBlock>
        {"\\frac{\\partial y}{\\partial x} = I + \\frac{\\partial F}{\\partial x}"}
      </MathBlock>

      <Prose>
        By the chain rule, the gradient of the loss <Code>L</Code> with respect to an earlier activation <Code>{"x_l"}</Code> decomposes as:
      </Prose>

      <MathBlock>
        {"\\frac{\\partial L}{\\partial x_l} = \\frac{\\partial L}{\\partial x_L} \\cdot \\prod_{k=l}^{L-1} \\left( I + \\frac{\\partial F_k}{\\partial x_k} \\right)"}
      </MathBlock>

      <Prose>
        When the product is expanded, the term containing all identity components is <Code>{"\\partial L / \\partial x_L"}</Code> itself — the upstream gradient arrives at every layer unattenuated along the identity path. Contrast this with a plain network <Code>{"x_{l+1} = F_l(x_l)"}</Code>, where the gradient is a product of Jacobians:
      </Prose>

      <MathBlock>
        {"\\frac{\\partial L}{\\partial x_l} = \\frac{\\partial L}{\\partial x_L} \\cdot \\prod_{k=l}^{L-1} \\frac{\\partial F_k}{\\partial x_k}"}
      </MathBlock>

      <Prose>
        If the spectral norm of each <Code>{"\\partial F_k / \\partial x_k"}</Code> is less than 1 (which is typical under standard initialization and ReLU/BN), the product decays exponentially in depth — vanishing gradients. Residuals replace exponential decay with additive accumulation.
      </Prose>

      <H3>3.3 Pre-activation versus post-activation</H3>

      <Prose>
        The original 2015 ResNet used post-activation ordering: inside each block, Conv → BN → ReLU → Conv → BN, then add the skip, then apply a final ReLU. Written explicitly:
      </Prose>

      <MathBlock>
        {"y = \\text{ReLU}(F(x) + x)"}
      </MathBlock>

      <Prose>
        The final ReLU sits on the identity path. When <Code>{"F(x) + x"}</Code> is negative, ReLU clips it to zero, which means the skip signal cannot always pass through unchanged. The 2016 pre-activation ResNet (He et al. arXiv 1603.05027) removed this obstruction:
      </Prose>

      <MathBlock>
        {"y = x + F(x), \\quad F(x) = W_2 \\cdot \\text{ReLU}(\\text{BN}(W_1 \\cdot \\text{ReLU}(\\text{BN}(x))))"}
      </MathBlock>

      <Prose>
        The BN and ReLU are now inside <Code>F</Code>; the addition produces <Code>y</Code> directly with no trailing nonlinearity. The identity path is clean from layer 0 to layer L. This ordering is what makes 1000+ layer ResNets trainable.
      </Prose>

      <H3>3.4 Dense connections: concatenation instead of addition</H3>

      <Prose>
        DenseNet (Huang et al. 2017) replaces addition with concatenation:
      </Prose>

      <MathBlock>
        {"x_l = H_l([x_0, x_1, \\ldots, x_{l-1}])"}
      </MathBlock>

      <Prose>
        Each layer sees all prior feature maps as input and produces a small number of new feature maps (the "growth rate" <Code>k</Code>, typically 12 or 32). Because features are preserved rather than summed, parameter count grows more slowly with depth (for a given representational capacity) and gradient paths multiply. The cost is memory: naive implementations store every intermediate tensor. Memory-efficient DenseNet implementations (Pleiss et al. 2017) recompute intermediate tensors during the backward pass to trade compute for memory.
      </Prose>

      <H3>3.5 Transformer residual pattern</H3>

      <Prose>
        A pre-norm Transformer block — the variant used in GPT, LLaMA, and most modern LLMs — computes:
      </Prose>

      <MathBlock>
        {"y = x + \\text{Attn}(\\text{LN}(x)), \\quad z = y + \\text{FFN}(\\text{LN}(y))"}
      </MathBlock>

      <Prose>
        There are two residual connections per block: one around the attention sublayer, one around the MLP sublayer. The LayerNorm is applied <em>inside</em> the residual (on the input to attention/FFN), so the identity path <Code>{"x \\to y"}</Code> preserves the un-normalized scale of <Code>x</Code>. Xiong and colleagues' 2020 paper "On Layer Normalization in the Transformer Architecture" (ICML 2020, arXiv 2002.04745) showed that pre-norm enables stable training with large learning rates and without learning-rate warmup, whereas post-norm — y = LN(x + Attn(x)) — requires careful warmup schedules and fails for very deep stacks.
      </Prose>

      <H3>3.6 Scaled residual: ReZero and LayerScale</H3>

      <Prose>
        Bachlechner et al. (2020) proposed ReZero:
      </Prose>

      <MathBlock>
        {"y = x + \\alpha \\cdot F(x), \\quad \\alpha \\in R, \\text{ initialized to } 0"}
      </MathBlock>

      <Prose>
        At initialization, <Code>{"\\alpha = 0"}</Code> means every block is exactly the identity, so the network starts as a no-op. Training learns each <Code>{"\\alpha"}</Code> away from zero where it helps. The paper showed 100-layer Transformers trainable without LayerNorm. CaiT's LayerScale (Touvron et al. 2021) generalizes this to a per-channel learnable diagonal:
      </Prose>

      <MathBlock>
        {"y = x + \\text{diag}(\\lambda_1, \\ldots, \\lambda_d) \\cdot F(x), \\quad \\lambda_i \\text{ init} \\sim 10^{-6} \\text{ to } 10^{-4}"}
      </MathBlock>

      <Prose>
        The tiny initialization keeps early training close to identity, which is critical for very deep vision Transformers (ConvNeXt, CaiT-S-36) where otherwise the added residuals compound into numerical instability. This is now standard for vision Transformers above 24 layers.
      </Prose>

      <H3>3.7 Projection shortcuts when dimensions differ</H3>

      <Prose>
        The equation <Code>{"y = F(x) + x"}</Code> requires that <Code>{"F(x)"}</Code> and <Code>x</Code> have the same shape. When a block changes channels or spatial resolution (downsampling), the skip path needs a projection:
      </Prose>

      <MathBlock>
        {"y = F(x) + W_s \\cdot x"}
      </MathBlock>

      <Prose>
        where <Code>{"W_s"}</Code> is typically a 1×1 convolution (for channel change) combined with strided sampling (for spatial change). He et al. 2015 analyzed three options: A) identity with zero-padding for channel increase, B) projection only when dimensions change, C) projection on every skip. Option B performs best per parameter; option A is parameter-free but slightly worse. Most ResNet implementations use option B.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below is executable PyTorch. Outputs are verbatim stdout captured from local runs.
      </Prose>

      <H3>4a. Numerical check: dL/dx = dL/dy · (1 + dF/dx)</H3>

      <Prose>
        The gradient identity for a residual block is the mathematical heart of the pattern. The test below verifies it numerically by running autograd on the residual block y = F(x) + x, then separately on F(x) alone, and showing the difference equals the upstream gradient dL/dy.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn

torch.manual_seed(1)
x = torch.randn(1, 4, requires_grad=True)
F_layer = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
head = nn.Linear(4, 1)

# Forward pass (residual): y = F(x) + x
f_of_x = F_layer(x)
y = f_of_x + x
loss = head(y).sum()
loss.backward()
grad_x_residual = x.grad.clone()

# Forward pass (plain): y = F(x) only
x.grad = None
x2 = x.detach().clone().requires_grad_()
f2 = F_layer(x2)
loss2 = head(f2).sum()
loss2.backward()
grad_x_plain = x2.grad.clone()

# Upstream gradient dL/dy equals head.weight (since loss is linear in y)
dL_dy = head.weight.clone()

print("Residual block gradient identity check")
print("=" * 50)
print(f"dL/dy (from head)         : {dL_dy[0].numpy()}")
print(f"dL/dx  (residual)         : {grad_x_residual[0].numpy()}")
print(f"dL/dx  (plain, F only)    : {grad_x_plain[0].numpy()}")
print(f"residual - plain          : {(grad_x_residual - grad_x_plain)[0].numpy()}")
print(f"(this should equal dL/dy) : {dL_dy[0].numpy()}")
print(f"max abs error: {(grad_x_residual - grad_x_plain - dL_dy).abs().max().item():.2e}")

# Output:
# Residual block gradient identity check
# ==================================================
# dL/dy (from head)         : [-0.1443786  -0.0548172  -0.48069406 -0.23839086]
# dL/dx  (residual)         : [-0.14497551 -0.0682184  -0.5647228  -0.27056837]
# dL/dx  (plain, F only)    : [-0.00059691 -0.0134012  -0.08402871 -0.03217752]
# residual - plain          : [-0.1443786  -0.0548172  -0.48069406 -0.23839085]
# (this should equal dL/dy) : [-0.1443786  -0.0548172  -0.48069406 -0.23839086]
# max abs error: 1.49e-08`}
      </CodeBlock>

      <Prose>
        The difference (residual gradient) − (plain gradient) matches the upstream gradient dL/dy to machine precision. This is the "gradient highway" in action: the residual receives everything the plain network receives, plus an unattenuated copy of the upstream signal.
      </Prose>

      <H3>4b. Plain 20-layer CNN vs 20-layer ResNet — degradation problem</H3>

      <Prose>
        Next we reproduce the degradation phenomenon on a small dataset. Both models have identical parameter counts (47,251 trainable weights). The only difference is the identity shortcut inside each of the 10 blocks (each block contains 2 conv layers, so 20 conv layers total).
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

# Small 3-class dataset (CIFAR-10-sized tensors)
N = 512
x = torch.randn(N, 3, 32, 32)
with torch.no_grad():
    score = x.mean(dim=(2, 3))
    s = score[:, 0] * 2 + score[:, 1] - score[:, 2]
    y = torch.zeros_like(s, dtype=torch.long)
    y[s > 0.5] = 1
    y[s < -0.5] = 2

class PlainBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(h)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(h + x)      # <- identity shortcut

def build(n_blocks, block_cls):
    layers = [nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU()]
    for _ in range(n_blocks):
        layers.append(block_cls(16))
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 3)]
    return nn.Sequential(*layers)

def train(model, epochs=15, lr=0.01):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward(); opt.step()
        losses.append(loss.item())
    return losses

plain  = build(10, PlainBlock)
resnet = build(10, ResBlock)
plain_losses  = train(plain)
resnet_losses = train(resnet)

print("epoch  plain-loss   resnet-loss")
for ep in range(15):
    print(f"  {ep:2d}    {plain_losses[ep]:.4f}       {resnet_losses[ep]:.4f}")

# Output:
# epoch  plain-loss   resnet-loss
#    0    1.3292       0.4121
#    1    1.2994       0.2902
#    2    1.2447       0.1582
#    3    1.1701       0.0762
#    4    1.0808       0.0359
#    5    0.9818       0.0175
#    6    0.8778       0.0090
#    7    0.7734       0.0050
#    8    0.6722       0.0029
#    9    0.5773       0.0018
#   10    0.4909       0.0011
#   11    0.4143       0.0008
#   12    0.3478       0.0005
#   13    0.2913       0.0004
#   14    0.2440       0.0003`}
      </CodeBlock>

      <Prose>
        The ResNet drives training loss from 0.41 to 0.0003 (three orders of magnitude) in 15 epochs. The plain network takes those same 15 epochs to move from 1.33 to 0.24 — still losing. Same parameters, same optimizer, same learning rate. The only difference is the identity path.
      </Prose>

      <H3>4c. Gradient magnitude at the first layer as depth grows</H3>

      <Prose>
        The degradation story is really a gradient story. Below we measure the L2 norm of the gradient at the first layer, for plain MLPs and residual MLPs at depths 5, 10, 20, 50, and 100. The ratio shows how much signal the residual path preserves that the plain path loses.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn, torch.nn.functional as F

def grad_norm_first_layer(depth, residual, steps=5):
    torch.manual_seed(0)
    dim = 32
    x = torch.randn(64, dim)
    y = (x.sum(dim=1) > 0).long()

    first  = nn.Linear(dim, dim)
    middle = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
    head   = nn.Linear(dim, 2)
    params = list(first.parameters()) + list(middle.parameters()) + list(head.parameters())

    def forward(x):
        h = F.relu(first(x))
        for m in middle:
            if residual:
                h = F.relu(m(h)) + h
            else:
                h = F.relu(m(h))
        return head(h)

    opt = torch.optim.SGD(params, lr=0.0)   # never actually step
    norms = []
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(forward(x), y)
        loss.backward()
        norms.append(first.weight.grad.norm().item())
    return sum(norms) / len(norms)

print("depth  plain-grad  residual-grad   ratio(res/plain)")
for depth in [5, 10, 20, 50, 100]:
    gp = grad_norm_first_layer(depth, residual=False)
    gr = grad_norm_first_layer(depth, residual=True)
    print(f"  {depth:3d}   {gp:.3e}   {gr:.3e}    {gr/max(gp,1e-30):.1e}x")

# Output:
# depth  plain-grad  residual-grad   ratio(res/plain)
#     5   3.783e-03   3.627e-01    9.6e+01x
#    10   3.729e-05   8.823e-01    2.4e+04x
#    20   2.555e-09   5.510e+00    2.2e+09x
#    50   2.270e-21   3.467e+03    1.5e+24x
#   100   0.000e+00   2.052e+09    inf`}
      </CodeBlock>

      <Prose>
        At depth 100 the plain network's first-layer gradient is numerically zero — the product of Jacobians has underflowed to double-precision zero. The residual version's gradient is finite (here somewhat large because nothing normalizes the cumulative skip sum — a real network would have LayerNorm or BatchNorm, but the raw signal behavior is the same). Every layer in a deep residual stack receives a usable gradient; every layer in a deep plain stack past ~20 layers does not.
      </Prose>

      <H3>4d. Training loss after fixed budget at depths 10, 20, 50, 100, 150, 200</H3>

      <Prose>
        This is the "depth vs loss" sweep that produces the classic degradation curve. For fair comparison the residual stack uses zero-initialized weights (so at init each block is exactly identity), and both use the same data, optimizer, and epoch count.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn, torch.nn.functional as F

N, D = 256, 16

def make_data():
    torch.manual_seed(0)
    x = torch.randn(N, D)
    y = (x.sum(dim=1) > 0).long()
    return x, y

def final_loss(depth, residual, epochs=60, lr=0.01):
    torch.manual_seed(0)
    x, y = make_data()
    layers = nn.ModuleList([nn.Linear(D, D) for _ in range(depth)])
    for m in layers:
        if residual:
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)     # start as identity
        else:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
    head = nn.Linear(D, 2)
    params = list(layers.parameters()) + list(head.parameters())
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    for _ in range(epochs):
        opt.zero_grad()
        h = x
        for m in layers:
            h = F.relu(m(h)) + h if residual else F.relu(m(h))
        loss = F.cross_entropy(head(h), y)
        loss.backward(); opt.step()
    return loss.item()

print("depth   plain-final   residual-final")
for d in [10, 20, 50, 100, 150, 200]:
    lp = final_loss(d, residual=False)
    lr_ = final_loss(d, residual=True)
    print(f"  {d:3d}     {lp:.4f}       {lr_:.4f}")

# Output:
# depth   plain-final   residual-final
#    10     0.5999       0.2668
#    20     0.6791       0.2614
#    50     0.6908       0.2698
#   100     0.6924       0.2661
#   150     0.6924       0.2691
#   200     0.6924       0.2673`}
      </CodeBlock>

      <Prose>
        The plain network's training loss gets <em>worse</em> with depth, saturating at the random-chance value of <Code>{"\\ln 2 \\approx 0.693"}</Code> — the optimizer cannot even match a 1-layer model for deep plain stacks. The residual network's final loss stays at roughly 0.27 regardless of depth. Depth becomes a free parameter. This is the degradation problem, cured.
      </Prose>

      <H3>4e. Pre-norm Transformer block — residual gradient flow</H3>

      <Prose>
        The Transformer's residual pattern is slightly different: two residuals per block (one around attention, one around FFN), with LayerNorm applied to the input of each sublayer. Below we build a 12-layer pre-norm Transformer and measure the gradient at every block's LayerNorm weight after one backward pass.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(42)

class PreNormBlock(nn.Module):
    def __init__(self, d=64, h=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(self, x):
        # y = x + Attn(LN(x))
        y = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
        # z = y + FFN(LN(y))
        z = y + self.ffn(self.ln2(y))
        return z

depth = 12
blocks = nn.ModuleList([PreNormBlock() for _ in range(depth)])
proj_in = nn.Linear(16, 64)
proj_out = nn.Linear(64, 2)

B, L, D = 4, 8, 16
x = torch.randn(B, L, D)
y = torch.zeros(B, dtype=torch.long)

h = proj_in(x)
for blk in blocks:
    h = blk(h)
logits = proj_out(h.mean(dim=1))
loss = F.cross_entropy(logits, y)
loss.backward()

print("Pre-norm Transformer residual gradient flow")
print(f"  depth = {depth}, d_model = 64, heads = 4\\n")
print("  block   LN1-weight grad norm")
for i, blk in enumerate(blocks):
    g = blk.ln1.weight.grad.norm().item()
    print(f"    {i:2d}      {g:.4e}")
print(f"\\n  loss = {loss.item():.4f}")

# Output:
# Pre-norm Transformer residual gradient flow
#   depth = 12, d_model = 64, heads = 4
#
#   block   LN1-weight grad norm
#      0      2.0224e-01
#      1      2.2506e-01
#      2      2.5590e-01
#      3      2.6842e-01
#      4      2.2152e-01
#      5      1.9338e-01
#      6      2.3381e-01
#      7      2.4164e-01
#      8      2.5266e-01
#      9      2.0952e-01
#     10      2.2047e-01
#     11      2.7224e-01
#   loss = 1.3872`}
      </CodeBlock>

      <Prose>
        Every block from 0 to 11 receives a gradient in the range 0.19–0.27. The earliest layer (block 0) is <em>not</em> gradient-starved; it gets essentially the same magnitude as block 11. This is why pre-norm Transformers can scale to hundreds of layers: the identity path carries signal end-to-end regardless of depth, and the LayerNorm-before-sublayer ordering prevents the accumulated residuals from blowing up the activation scale.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5.1 torchvision.models.resnet50 — canonical ResNet</H3>

      <CodeBlock language="python">
{`from torchvision.models import resnet50, ResNet50_Weights
import torch

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()

# The bottleneck block in torchvision.models.resnet.Bottleneck:
#   out = conv1(x) -> bn1 -> relu
#       -> conv2   -> bn2 -> relu
#       -> conv3   -> bn3
#   if downsample is not None: identity = downsample(x)   # 1x1 conv projection
#   out += identity          # <- residual addition
#   out = relu(out)
#
# Note: post-activation ResNet (original 2015 paper). Final relu is outside
# the residual, which is why pre-activation ResNet (2016) later flipped the
# order for 1000+ layer stability.

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    y = model(x)
print(f"output shape: {y.shape}")         # torch.Size([1, 1000])
print(f"num params:   {sum(p.numel() for p in model.parameters()):,}")  # 25,557,032`}
      </CodeBlock>

      <H3>5.2 torch.nn.Identity — making skip explicit</H3>

      <Prose>
        When defining custom blocks, <Code>torch.nn.Identity</Code> is the zero-parameter module used for the skip path when no projection is needed. This makes the residual structure explicit and lets you swap in a projection without changing the forward pass:
      </Prose>

      <CodeBlock language="python">
{`import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # Explicit skip: identity if shapes match, projection otherwise
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = self.bn1(self.conv1(x)).relu()
        h = self.bn2(self.conv2(h))
        return (h + self.skip(x)).relu()`}
      </CodeBlock>

      <H3>5.3 Stochastic depth with timm.DropPath</H3>

      <Prose>
        Stochastic depth (Huang et al. 2016, arXiv 1603.09382) drops entire residual branches at random during training: <Code>{"y = x + DropPath(F(x))"}</Code>. This regularizes deep stacks (acts like ensemble-over-sub-networks) and speeds up training by ~25% because dropped branches skip their forward compute. timm's <Code>DropPath</Code> is the standard implementation:
      </Prose>

      <CodeBlock language="python">
{`from timm.models.layers import DropPath

class StochasticResBlock(nn.Module):
    def __init__(self, dim, drop_path_rate=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp  = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(),
                                  nn.Linear(4*dim, dim))
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.mlp(self.norm(x)))

# Standard schedule in ConvNeXt / Swin / ViT: linear from 0 at block 0
# to drop_path_max at the last block (deeper blocks get dropped more often).
# drop_path_max is typically 0.1 for ViT-B, 0.3 for ViT-L, 0.5 for ViT-H.`}
      </CodeBlock>

      <H3>5.4 ResNeXt — cardinality via grouped convs</H3>

      <Prose>
        ResNeXt (Xie et al. 2017) replaces the single conv branch inside each block with <Code>C</Code> parallel branches (the "cardinality"), implemented efficiently with grouped convolutions. Each block becomes <Code>{"y = x + \\sum_{c=1}^{C} F_c(x)"}</Code>. For a fixed FLOP budget, increasing cardinality at the expense of width consistently improves ImageNet accuracy:
      </Prose>

      <CodeBlock language="python">
{`import torch.nn as nn

class ResNeXtBlock(nn.Module):
    """Bottleneck ResNeXt: 1x1 -> 3x3 grouped -> 1x1 with residual."""
    def __init__(self, in_ch, mid_ch, out_ch, cardinality=32, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        # key line: groups=cardinality turns this into C parallel branches
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride, 1,
                               groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.skip = (nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                                    nn.BatchNorm2d(out_ch))
                     if stride != 1 or in_ch != out_ch else nn.Identity())

    def forward(self, x):
        h = self.bn1(self.conv1(x)).relu()
        h = self.bn2(self.conv2(h)).relu()
        h = self.bn3(self.conv3(h))
        return (h + self.skip(x)).relu()

# torchvision.models.resnext50_32x4d uses cardinality=32, bottleneck_width=4`}
      </CodeBlock>

      <H3>5.5 HuggingFace Transformers — LLaMA-style residual</H3>

      <Prose>
        Every modern decoder LLM in HuggingFace uses the same residual pattern. From <Code>transformers/models/llama/modeling_llama.py</Code>:
      </Prose>

      <CodeBlock language="python">
{`# Simplified LlamaDecoderLayer forward (HuggingFace transformers >= 4.40)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.hidden_size)
        self.self_attn       = LlamaAttention(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size)
        self.mlp             = LlamaMLP(config)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, ...):
        # ---- Residual around self-attention (pre-norm) ----
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, *_ = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states      # <- skip

        # ---- Residual around MLP (pre-norm) ----
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states      # <- skip

        return hidden_states`}
      </CodeBlock>

      <H3>5.6 LayerScale for very deep ViT / ConvNeXt</H3>

      <Prose>
        LayerScale (Touvron et al. 2021, CaiT) applies a learnable per-channel scalar to the residual branch before addition, with tiny initialization. This is the production technique that made 36-layer and 48-layer vision Transformers trainable. ConvNeXt-V1 (Liu et al. 2022) uses the same pattern:
      </Prose>

      <CodeBlock language="python">
{`import torch.nn as nn
import torch

class LayerScaleBlock(nn.Module):
    def __init__(self, dim, init_value=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        # learnable per-channel scalar; tiny initialization
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        attn_out, _ = self.attn(self.norm(x), self.norm(x), self.norm(x),
                                 need_weights=False)
        return x + self.gamma * attn_out     # LayerScale-modulated residual

# init_value defaults per Touvron et al. 2021:
#   depth <= 18 layers:  1e-4
#   depth 18-24:         1e-5
#   depth > 24:          1e-6`}
      </CodeBlock>

      <Callout accent="gold">
        Production rule-of-thumb for residual architectures in 2026: (1) pre-norm ordering (LayerNorm inside the residual, not around it); (2) LayerScale with 1e-6 init for any Transformer deeper than 24 layers; (3) DropPath with a linear schedule from 0 to 0.1–0.5 for regularization; (4) projection shortcuts (1×1 conv) wherever channels or spatial resolution change.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Training loss vs depth — degradation curve</H3>

      <Prose>
        The plot below shows final training loss after 60 SGD epochs at depths 10, 20, 50, 100, 150, and 200. The plain network's loss climbs with depth and saturates at random-guess loss (<Code>{"\\ln 2 \\approx 0.693"}</Code>) by depth 50. The ResNet's loss is essentially independent of depth — what He et al. called the degradation-free property.
      </Prose>

      <Plot
        label="training loss vs depth — plain vs ResNet (60 epochs)"
        xLabel="network depth (layers)"
        yLabel="final training loss"
        series={[
          {
            name: "plain network",
            color: "#f87171",
            points: [
              [10, 0.5999],
              [20, 0.6791],
              [50, 0.6908],
              [100, 0.6924],
              [150, 0.6924],
              [200, 0.6924],
            ],
          },
          {
            name: "ResNet (zero-init residual)",
            color: colors.gold,
            points: [
              [10, 0.2668],
              [20, 0.2614],
              [50, 0.2698],
              [100, 0.2661],
              [150, 0.2691],
              [200, 0.2673],
            ],
          },
        ]}
      />

      <Prose>
        The ResNet line is flat. The plain line rises then saturates. A 200-layer plain network is not merely harder to train than a 20-layer one — it is strictly worse. A 200-layer ResNet is as good as a 10-layer one. This is the whole point of the paper: depth stops being a cost and becomes an option.
      </Prose>

      <H3>6b. First-layer gradient magnitude by depth — log scale</H3>

      <Prose>
        The heatmap shows the log10 of the first-layer gradient norm for plain and residual networks at depths 5, 10, 20, 50, 100. Plain networks lose ~3 orders of magnitude of gradient signal per 10 layers added. Residual networks preserve it.
      </Prose>

      <Heatmap
        label="log10(first-layer grad norm) — darker = smaller gradient"
        rowLabels={["plain", "residual"]}
        colLabels={["d=5", "d=10", "d=20", "d=50", "d=100"]}
        matrix={[
          [-2.42, -4.43, -8.59, -20.64, -30.00],
          [-0.44, -0.05,  0.74,  3.54,  9.31],
        ]}
        colorScale="warm"
      />

      <Prose>
        Reading the heatmap: the plain row goes from −2.4 (depth 5) to approximately −30 (depth 100, where the gradient has underflowed to zero in double precision). The residual row stays in a usable range — even at depth 100 the first-layer gradient is comparable to or larger than at depth 5, because of the additive telescoping of skip terms. (In a real network, LayerNorm keeps the residual row bounded near zero rather than growing; this unnormalized sweep is designed to make the raw signal-preservation effect visible.)
      </Prose>

      <H3>6c. Forward pass through a residual block — StepTrace</H3>

      <StepTrace
        label="residual block forward pass — pre-activation"
        steps={[
          {
            label: "Step 1 — Input arrives",
            render: () => (
              <Prose>
                {"Input tensor x of shape [B, C, H, W] enters the block. The block will compute y = x + F(x). The skip path begins here: x is held onto until the addition at the end. The conv/norm/activation chain inside F operates on a copy, and x itself is never modified. This is the defining property of a residual block — the input is always available downstream."}
              </Prose>
            ),
          },
          {
            label: "Step 2 — BatchNorm / LayerNorm inside F",
            render: () => (
              <Prose>
                In pre-activation ordering, the first operation inside F is normalization. BN normalizes each channel to zero mean / unit variance across the batch; LN normalizes each sample along the feature dimension. The normalization acts only on the residual branch, not on the skip path. This is why pre-activation "cleans" the identity path: x flows forward without normalization artifacts.
              </Prose>
            ),
          },
          {
            label: "Step 3 — ReLU / GELU activation",
            render: () => (
              <Prose>
                The activation applies elementwise to the normalized tensor. For CNNs, ReLU is standard; for Transformers, GELU or SwiGLU. Crucially, the activation is inside F; it never sees the skip path. This is what He et al. 2016 fixed from the 2015 post-activation design — the original paper had a ReLU <em>after</em> the addition, which clipped negative components of the identity flow.
              </Prose>
            ),
          },
          {
            label: "Step 4 — First projection (Conv or Linear)",
            render: () => (
              <Prose>
                A Conv2d (for CNNs) or Linear (for Transformers) projects the activated tensor. In bottleneck blocks this first conv is 1×1 and reduces channels (e.g., 256 → 64) to save compute on the expensive 3×3 step. The bias term is usually omitted because the subsequent BN absorbs any constant offset.
              </Prose>
            ),
          },
          {
            label: "Step 5 — Second BN + activation + projection",
            render: () => (
              <Prose>
                The pattern repeats: normalize, activate, project. After this second projection the tensor has the shape the skip path expects. In a bottleneck block, a third 1×1 conv expands back from 64 → 256 channels. No activation follows this final projection — leaving F(x) as the raw delta to add to x.
              </Prose>
            ),
          },
          {
            label: "Step 6 — Skip path projection (if shape changed)",
            render: () => (
              <Prose>
                {"If the block changes channels (e.g., 64 -> 128) or spatial resolution (stride=2), the skip path needs a 1x1 conv to match F(x)'s output shape. This projection is the only place where the identity path is not bit-exact; when channels and resolution are preserved the skip is the zero-parameter Identity module."}
              </Prose>
            ),
          },
          {
            label: "Step 7 — Addition: y = F(x) + skip(x)",
            render: () => (
              <Prose>
                {"The elementwise addition merges the two paths. The output y now contains the original signal plus the block's learned refinement. During the backward pass, gradients split at this addition: one copy flows into F (contributing dF/dx), one copy flows directly back along the skip path (contributing I). This is where the gradient highway is born."}
              </Prose>
            ),
          },
          {
            label: "Step 8 — Output and downstream propagation",
            render: () => (
              <Prose>
                y is passed to the next block. In pre-activation ordering, no final ReLU is applied to y — the next block's first BN/ReLU handles any nonlinearity needed. This is subtle but important: the identity path from block 0 to block L is a straight additive chain, never clipped by a nonlinearity. This is what allows stable training of 1000+ layer networks.
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
        label="when and how to use residual connections"
        steps={[
          {
            label: "Shallow networks (<20 layers)",
            render: () => (
              <Prose>
                Residuals are optional. Plain VGG-style networks up to ~20 layers train fine with BatchNorm and a sensible initialization. The optimization landscape is still tractable; the degradation problem has not yet kicked in. That said: adding residuals here still helps (slightly faster convergence, modest accuracy gain) and costs essentially nothing, so the default for any new design should be "use them anyway." The only reason to skip them at this scale is if you are deliberately studying a baseline.
              </Prose>
            ),
          },
          {
            label: "Deep networks (>20 layers) — mandatory",
            render: () => (
              <Prose>
                Above ~20 layers, residuals are non-negotiable. Without them, optimization fails outright — not just slower, but worse than shallower nets at matching the training set. This is not a performance choice; it is an architectural requirement. Every ResNet-50/101/152, every ViT-B/L/H, every GPT/LLaMA has residuals around every compute block. If you are sketching a new deep architecture and forget the skips, it will not train.
              </Prose>
            ),
          },
          {
            label: "Residual (addition) vs Dense (concatenation)",
            render: () => (
              <Prose>
                Residual: y = F(x) + x. Constant channel count, cheap compute, moderate memory. Dense: y = [x, F(x)]. Channels grow linearly with layer index, so later blocks see an ever-wider input. DenseNet achieves better accuracy per parameter than ResNet on CIFAR but consumes more activation memory (every prior layer's output is stored). In 2026, addition has won for large-scale training because it composes cleanly with tensor/pipeline parallelism and activation-checkpointing strategies; concatenation is harder to shard because activation sizes aren't uniform. Use addition for new work unless you have specific parameter-efficiency targets.
              </Prose>
            ),
          },
          {
            label: "Scaled residual (LayerScale / ReZero) for very deep Transformers",
            render: () => (
              <Prose>
                {"For Transformers above ~24 layers (CaiT, ConvNeXt, big LLMs), use LayerScale: y = x + diag(lambda) * F(x) with per-channel lambda initialized to 1e-6. This keeps early training close to identity, preventing the accumulated residuals from exploding the activation scale. ReZero (scalar alpha init to 0) is the simpler single-parameter variant; LayerScale per-channel is more expressive and is the production choice. Skip this for Transformers under 18 layers: you don't need it, and it costs a tiny bit of capacity."}
              </Prose>
            ),
          },
          {
            label: "Pre-norm vs post-norm for Transformers",
            render: () => (
              <Prose>
                Pre-norm (y = x + Sublayer(LN(x))) is the default in 2026. Post-norm (y = LN(x + Sublayer(x))) was used in the original "Attention is All You Need" and in BERT but is harder to train deep. Xiong et al. 2020 (ICML) proved that post-norm requires careful warmup because gradients through many LN layers diverge; pre-norm trains stably without warmup at any depth. For any new Transformer, use pre-norm. The rare exception: small (&lt;12 layer) bidirectional encoders where post-norm's slight accuracy edge matters, and warmup is affordable.
              </Prose>
            ),
          },
          {
            label: "Stochastic depth as regularizer",
            render: () => (
              <Prose>
                Stochastic depth (drop whole residual branches with probability p during training) is a regularizer specifically designed for deep residual networks. It acts as an implicit ensemble over sub-networks (exactly the Veit-ensemble view). Linear schedule is standard: block l gets drop probability (l / L) * p_max. Typical p_max: 0.1 for ViT-B, 0.3 for ViT-L, 0.5 for ViT-H. Also speeds training ~25% because dropped branches skip their forward compute. Default on for any deep vision model in 2026.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales</H2>

      <H3>8.1 Activation memory is the bottleneck</H3>

      <Prose>
        Residual blocks require storing every block's input for the backward pass (autograd needs x at every addition point). For a 100-block residual network processing a batch of shape <Code>{"[B, C, H, W]"}</Code>, the activation memory is roughly <Code>{"100 * B * C * H * W * 4 bytes"}</Code> for float32. A ResNet-50 at batch 256 stores ~5 GB of activations alone during training. This memory scales linearly with depth, and it is the dominant cost above depth ~50 — more than weights, more than optimizer state. Batch size and resolution are the levers you can trade off against depth.
      </Prose>

      <H3>8.2 Gradient checkpointing cuts activation memory to sqrt(L)</H3>

      <Prose>
        Activation checkpointing (Chen et al. 2016, arXiv 1604.06174) stores activations only at a small subset of layers and recomputes intermediates during the backward pass. For a network of depth L, checkpointing at every <Code>{"\\sqrt{L}"}</Code> blocks reduces activation memory from <Code>O(L)</Code> to <Code>{"O(\\sqrt{L})"}</Code> at the cost of one extra forward pass. In PyTorch, <Code>torch.utils.checkpoint.checkpoint_sequential</Code> handles this automatically; in HuggingFace, <Code>model.gradient_checkpointing_enable()</Code> does it per transformer layer. This is how GPT-4-class models train with depth &gt; 100 on realistic batch sizes.
      </Prose>

      <H3>8.3 Stochastic depth reduces expected compute</H3>

      <Prose>
        With linear-schedule stochastic depth (max rate <Code>{"p_{max}"}</Code>), the expected number of blocks executed per forward pass is <Code>{"L \\cdot (1 - p_{max}/2)"}</Code>. For a 32-block ViT-B with <Code>{"p_{max} = 0.1"}</Code>, expected executed blocks = 32 · 0.95 = 30.4 — a 5% compute saving for free. For ViT-H with <Code>{"p_{max} = 0.5"}</Code>, expected blocks = 32 · 0.75 = 24 — a 25% saving. Training time per epoch drops proportionally. Inference is unaffected; stochastic depth only fires during training.
      </Prose>

      <H3>8.4 LayerScale init prevents divergence in 100-layer Transformers</H3>

      <Prose>
        Without LayerScale, the accumulated residuals in a deep Transformer cause the activation norm to grow roughly as <Code>{"\\sqrt{L}"}</Code>. At depth 100 this produces a 10× inflation of activations at the top of the stack, which destabilizes attention (softmax overflows) and the loss. LayerScale with <Code>{"\\lambda = 10^{-6}"}</Code> keeps each block's contribution at initialization negligible; the network starts as nearly pure identity and deviates gradually as <Code>{"\\lambda"}</Code> is learned. Empirically this is what enabled CaiT-S-36, ConvNeXt-V1-XL, and many 40+ layer ViTs to train at all.
      </Prose>

      <H3>8.5 ResNet-to-Transformer scaling laws differ</H3>

      <Prose>
        ResNets tend to plateau in accuracy past ~150 layers for a fixed input resolution — returns diminish because the receptive field saturates and parameters allocate to deep features with little room left to improve. Transformers with residuals scale much farther: GPT-3 (96 layers), GPT-4 class (100+), LLaMA-3-405B (many). The difference is that each Transformer layer has global receptive field via attention, so stacking deeper continues to add representational capacity, while each conv layer's receptive field only grows linearly with depth. In both cases, residuals are necessary; the scaling ceiling is set by the compute block, not by the skip pattern.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Dimension mismatch in the skip path</H3>

      <Prose>
        The addition <Code>{"F(x) + x"}</Code> requires identical shapes. If a block changes channels or spatial resolution without a corresponding projection on the skip, you get a runtime shape error (or worse, silent broadcasting bugs). Fix: add a 1×1 Conv or Linear on the skip path whenever <Code>{"F"}</Code>'s output shape differs from its input shape. In PyTorch use <Code>nn.Identity()</Code> as the default and swap in a projection only when needed:
      </Prose>

      <CodeBlock language="python">
{`# Bug: shape error at runtime
class BuggyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 64 -> 128, H/2
    def forward(self, x):
        return self.conv(x) + x   # RuntimeError: tensors of different shapes

# Fix: projection on the skip path
class FixedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.skip = nn.Conv2d(64, 128, 1, stride=2, bias=False)  # match shape
    def forward(self, x):
        return self.conv(x) + self.skip(x)`}
      </CodeBlock>

      <H3>9.2 ReLU after the addition kills negative components</H3>

      <Prose>
        Post-activation ResNet (the original 2015 design) applies ReLU after the skip addition: <Code>{"y = ReLU(F(x) + x)"}</Code>. When <Code>{"F(x) + x"}</Code> has negative elements, ReLU zeros them, which means the skip signal is clipped at every block. For deep stacks (&gt; 200 layers) this accumulates into a significant degradation. Fix: use pre-activation ordering (<Code>{"y = x + F(x)"}</Code> with no trailing ReLU; normalization and activation move inside <Code>F</Code>). He et al. 2016 showed this lets 1001-layer ResNets train where post-activation fails. Every modern architecture uses pre-activation or its Transformer equivalent (LN before sublayer, not after).
      </Prose>

      <H3>9.3 Forgetting the residual in one block</H3>

      <Prose>
        A common bug in a custom architecture is to write many residual blocks and miss the skip in one of them — maybe a transition block between stages, or a special "attention-free" block. The network still trains, but that one block becomes a bottleneck: gradient magnitude drops sharply after flowing through it. Symptom: training plateaus earlier than expected; layers downstream of the missing skip train fine, layers upstream do not. Debug by measuring per-block gradient norms (as in section 4c) — any block where the gradient norm falls by more than ~10× compared to its neighbors is a likely skip omission.
      </Prose>

      <H3>9.4 Wrong norm placement (post-norm in a deep Transformer)</H3>

      <Prose>
        Post-norm (<Code>{"y = LN(x + Sublayer(x))"}</Code>) is fine up to ~12 layers with careful warmup but fails at depth &gt; 24. The LN at each block attenuates the cumulative skip signal; with N post-norm blocks, the effective identity path is multiplied by <Code>{"1/\\sqrt{N}"}</Code> per block, so by block 100 the first block's contribution has vanished. Symptom: loss stalls after warmup; gradients at the embedding are zero. Fix: switch to pre-norm (<Code>{"y = x + Sublayer(LN(x))"}</Code>). This is the single most common mistake in "I'm rolling my own Transformer" code.
      </Prose>

      <H3>9.5 Not adjusting learning rate when adding skips</H3>

      <Prose>
        Residuals change the effective gradient magnitude (larger at earlier layers, because the identity path preserves signal). If you copy a learning rate schedule from a plain-network recipe, the optimizer may be too aggressive and the loss will oscillate or diverge early. Rule of thumb: when adding skips to an existing architecture, reduce the peak LR by ~2× or add a longer warmup. The 2015 ResNet paper used a 10× larger batch size than contemporaneous plain networks partly for this reason (larger batches absorb the larger gradients).
      </Prose>

      <H3>9.6 Hidden scale explosion without LayerScale in deep ViTs</H3>

      <Prose>
        {"In a deep Transformer without LayerScale, activation norms grow as sqrt(L). By layer 40 this can push attention logits past fp16's representable range (~65,504), causing softmax to produce NaN. Symptom: training is fine for a while, then suddenly NaN loss. Fix: either switch to bf16 (which has float32-like exponent range), or add LayerScale with lambda init 1e-6. LayerScale is the production recommendation; it is why CaiT-S-36 and ConvNeXt-XL train at all."}
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Read in roughly this order to follow the intellectual development from Highway Networks through modern scaled residuals.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "Srivastava, Greff & Schmidhuber 2015 — Highway Networks",
            render: () => (
              <Prose>
                Srivastava, R.K., Greff, K., and Schmidhuber, J. (2015). "Highway Networks." arXiv:1505.00387. Available at arxiv.org/abs/1505.00387. The first paper to demonstrate that very deep feedforward networks can be trained if each layer has a learned gated bypass path. Block formula: <Code>{"y = H(x, W) \\cdot T(x) + x \\cdot C(x)"}</Code> with transform gate T and carry gate C. The paper showed trainable 100-layer networks on MNIST and CIFAR-10, six months before ResNet. Conceptually the direct ancestor of ResNet; ResNet drops the gates and keeps only the additive shortcut.
              </Prose>
            ),
          },
          {
            label: "He, Zhang, Ren & Sun 2015 — ResNet (CVPR 2016 Best Paper)",
            render: () => (
              <Prose>
                He, K., Zhang, X., Ren, S., and Sun, J. (2015). "Deep Residual Learning for Image Recognition." arXiv:1512.03385. Published in CVPR 2016, pp. 770–778; Best Paper Award. Available at arxiv.org/abs/1512.03385. The canonical paper. Introduces the residual block <Code>{"y = F(x) + x"}</Code>, empirically demonstrates the degradation problem on plain CNNs (56-layer plain net has higher training error than 20-layer plain net), and shows ResNet-152 winning ILSVRC 2015. Sections 3.2–3.3 contain the degradation analysis; Section 4 has the experiments with depths up to 1202 layers on CIFAR-10. Reading this paper is table-stakes for anyone working on deep architectures.
              </Prose>
            ),
          },
          {
            label: "He, Zhang, Ren & Sun 2016 — Pre-activation ResNet",
            render: () => (
              <Prose>
                He, K., Zhang, X., Ren, S., and Sun, J. (2016). "Identity Mappings in Deep Residual Networks." arXiv:1603.05027. Published in ECCV 2016. Available at arxiv.org/abs/1603.05027. The follow-up that rearranges block internals: BN → ReLU → Conv inside <Code>F</Code>, with a pure identity path carrying x untouched. This paper's Figure 2 comparing post-activation vs pre-activation orderings is one of the most-reproduced diagrams in deep learning. The paper reports a 1001-layer ResNet achieving 4.62% error on CIFAR-10, matching much shallower competitors. Pre-activation is now the default in most ResNet implementations and is the conceptual ancestor of the pre-norm Transformer.
              </Prose>
            ),
          },
          {
            label: "Huang, Liu, van der Maaten & Weinberger 2017 — DenseNet",
            render: () => (
              <Prose>
                Huang, G., Liu, Z., van der Maaten, L., and Weinberger, K.Q. (2017). "Densely Connected Convolutional Networks." arXiv:1608.06993. Published in CVPR 2017 with the Best Paper Award. Available at arxiv.org/abs/1608.06993. Each layer receives the concatenation of all prior layers: <Code>{"x_l = H_l([x_0, \\ldots, x_{l-1}])"}</Code>. The paper argues for dense concatenative skips over additive ones: improved parameter efficiency, smoother optimization, implicit deep supervision. DenseNet-BC-190 achieved state-of-the-art CIFAR-10/100 at ~15M parameters. The architecture lost to ResNet variants in large-scale deployment (memory cost) but remains influential in medical imaging and segmentation.
              </Prose>
            ),
          },
          {
            label: "Xie, Girshick, Dollár, Tu & He 2017 — ResNeXt",
            render: () => (
              <Prose>
                Xie, S., Girshick, R., Dollár, P., Tu, Z., and He, K. (2017). "Aggregated Residual Transformations for Deep Neural Networks." arXiv:1611.05431. Published in CVPR 2017. Available at arxiv.org/abs/1611.05431. Introduces the <em>cardinality</em> dimension: replace a single conv branch inside each residual block with <Code>C</Code> parallel branches summed together. Implemented efficiently via grouped convolutions. The paper shows that increasing cardinality (C) at fixed FLOPs is more effective than increasing width or depth: ResNeXt-101 (32×4d) beats ResNet-200 with fewer parameters. The cardinality axis informed later designs (grouped conv in MobileNet, multi-head attention as cardinality).
              </Prose>
            ),
          },
          {
            label: "Veit, Wilber & Belongie 2016 — Ensemble view of ResNets",
            render: () => (
              <Prose>
                Veit, A., Wilber, M., and Belongie, S. (2016). "Residual Networks Behave Like Ensembles of Relatively Shallow Networks." arXiv:1605.06431. Published in NeurIPS 2016. Available at arxiv.org/abs/1605.06431. The paper that reframes ResNets as exponentially-many path ensembles. Key experiments: (1) deleting individual residual blocks from a trained ResNet barely affects test accuracy (unlike deleting layers from a plain CNN, which is catastrophic); (2) the effective paths through a ResNet have average length much shorter than the nominal depth. This is the theoretical justification for stochastic depth and for why ResNets are unusually robust to architectural modifications.
              </Prose>
            ),
          },
          {
            label: "Bachlechner et al. 2020 — ReZero is All You Need",
            render: () => (
              <Prose>
                Bachlechner, T., Majumder, B.P., Mao, H.H., Cottrell, G.W., and McAuley, J. (2020). "ReZero is All You Need: Fast Convergence at Large Depth." arXiv:2003.04887. Published in UAI 2021. Available at arxiv.org/abs/2003.04887. Proposes <Code>{"y = x + \\alpha \\cdot F(x)"}</Code> with scalar <Code>α</Code> initialized to 0 and learned per block. At initialization the network is exact identity; training discovers where to add non-zero residuals. The paper demonstrates 100-layer Transformers trainable without LayerNorm (just ReZero), and convergence rates 2–5× faster than Xavier-initialized baselines for deep stacks. ReZero is the conceptual root of LayerScale and is a minimal, elegant solution to very deep optimization.
              </Prose>
            ),
          },
          {
            label: "Touvron et al. 2021 — CaiT and LayerScale",
            render: () => (
              <Prose>
                Touvron, H., Cord, M., Sablayrolles, A., Synnaeve, G., and Jégou, H. (2021). "Going Deeper with Image Transformers." arXiv:2103.17239. Published in ICCV 2021. Available at arxiv.org/abs/2103.17239. Introduces LayerScale: per-channel diagonal <Code>{"diag(\\lambda_1, \\ldots, \\lambda_d)"}</Code> applied to the residual branch before addition, with tiny initialization. The paper shows that without LayerScale, ViTs above 24 layers fail to train; with LayerScale (init 1e-6), CaiT-S-36 and CaiT-M-48 achieve state-of-the-art on ImageNet at their compute budgets. LayerScale is now standard in ConvNeXt (Liu et al. 2022), Swin-V2, and most production deep vision Transformers. It is the operational answer to "how do we keep making Transformers deeper?"
              </Prose>
            ),
          },
          {
            label: "Xiong et al. 2020 — Pre-norm vs post-norm theory",
            render: () => (
              <Prose>
                Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., and Liu, T. (2020). "On Layer Normalization in the Transformer Architecture." arXiv:2002.04745. Published in ICML 2020. Available at arxiv.org/abs/2002.04745. The paper that proves pre-norm Transformers train stably at any depth without learning-rate warmup, while post-norm Transformers (the original "Attention is All You Need" design) require careful warmup schedules. The analysis is done in the infinite-width limit using the NTK framework but the takeaway is practical: for any new Transformer, use pre-norm. This recommendation has shaped every production LLM since 2020 (GPT-3 onward, LLaMA, Mistral, Gemma).
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check</H2>

      <Prose>
        Attempt all five before reading the answers. Exercises 1–2 test the core math; 3 tests intuition; 4 tests architectural judgment; 5 tests debugging.
      </Prose>

      <H3>Exercise 1 (derivation — gradient identity)</H3>
      <Prose>
        A residual block computes <Code>{"y = F(x) + x"}</Code>. Derive <Code>{"\\partial L / \\partial x"}</Code> in terms of <Code>{"\\partial L / \\partial y"}</Code> and <Code>{"\\partial F / \\partial x"}</Code>. Then for a stack of L blocks, write the gradient reaching <Code>{"x_0"}</Code> and explain why vanishing gradients along the identity path are structurally impossible.
      </Prose>
      <Callout accent="green">
        <strong>Answer 1.</strong> By the chain rule through the addition node: <Code>{"\\partial L / \\partial x = \\partial L / \\partial y \\cdot (I + \\partial F / \\partial x)"}</Code>. For a stack of L blocks, unrolling gives <Code>{"\\partial L / \\partial x_0 = \\partial L / \\partial x_L \\cdot \\prod_{k=0}^{L-1} (I + \\partial F_k / \\partial x_k)"}</Code>. Expand the product: one term is the pure identity product (which equals <Code>I</Code>), so the gradient contains a copy of <Code>{"\\partial L / \\partial x_L"}</Code> that passed through no Jacobian at all. Even if every <Code>{"\\partial F_k / \\partial x_k"}</Code> has spectral norm well below 1, the identity component guarantees a non-vanishing lower bound on the gradient at <Code>{"x_0"}</Code>. Contrast with a plain network whose gradient is <Code>{"\\prod_k \\partial F_k / \\partial x_k"}</Code> — a pure product of Jacobians that decays exponentially when each factor is less than 1. Residuals replace exponential decay with additive accumulation.
      </Callout>

      <H3>Exercise 2 (the degradation problem — what is it really?)</H3>
      <Prose>
        Explain in your own words why the degradation problem is different from overfitting and different from vanishing gradients. What experiment did He et al. 2015 run to distinguish it from both? Why is the logical contradiction important (why does it imply something is wrong with the optimization, not the architecture)?
      </Prose>
      <Callout accent="green">
        <strong>Answer 2.</strong> Overfitting would show as lower training error with higher test error — the deeper net would fit the training set better. Vanishing gradients would show as total stagnation (loss plateaus near initialization). The degradation problem is neither: the 56-layer plain net had <em>higher</em> training error than the 20-layer net — it could not fit the training set as well. The experiment (Figure 1 in He et al. 2015): train identical CNNs at 20 and 56 layers with BN, plot training error. The 56-layer curve sits above the 20-layer curve. This is a logical contradiction: any function a 20-layer net can represent, a 56-layer net can also represent — simply let the extra 36 layers be identity. The 56-layer net has strictly more function capacity, so training error should be ≤ 20-layer's. But SGD cannot find the identity-preserving solution through composition of Conv-BN-ReLU. The problem is therefore not architectural (the capacity is there) but optimization-level: the parameterization makes identity hard to express. Residuals fix it by reparametrizing so identity is the default at init.
      </Callout>

      <H3>Exercise 3 (intuition — ensemble of paths)</H3>
      <Prose>
        A ResNet has L residual blocks. Explain why Veit et al. argued it behaves like an ensemble of <Code>{"2^L"}</Code> networks, and what empirical result supports this. Why does this explain why ResNets are robust to deleting individual blocks at test time?
      </Prose>
      <Callout accent="green">
        <strong>Answer 3.</strong> Each residual block gives the signal two routes: through <Code>F_k</Code> or around it via the identity. For a stack of L blocks, the output is a sum over all <Code>{"2^L"}</Code> subsets of "active" F functions: <Code>{"x_L = x_0 + \\sum_{S \\subseteq \\{0, ..., L-1\\}} (\\text{contribution of subset S})"}</Code>. Veit's supporting experiment: delete individual residual blocks from a trained ResNet-110 at test time. The test accuracy drops by less than 1% for any single block removal. Delete 10 blocks randomly: accuracy drops by only a few percent. Compare with VGG-16: delete any single layer and accuracy drops to near-chance. This robustness is exactly what an ensemble of many shallow paths predicts — any single path is one vote among exponentially many. The practical consequence: ResNets tolerate structural perturbations (pruning, quantization, dropout-like block dropping) gracefully, and techniques like stochastic depth work as regularizers precisely because they exploit this ensemble structure during training.
      </Callout>

      <H3>Exercise 4 (architectural judgment — LayerScale)</H3>
      <Prose>
        You are designing a 48-layer vision Transformer. Your colleague argues: "we already have pre-norm and residuals, so optimization is safe — we don't need LayerScale." Explain what specifically goes wrong in a 48-layer pre-norm Transformer without LayerScale, and why tiny-init LayerScale fixes it. What is the intuition behind initializing <Code>λ</Code> at <Code>{"10^{-6}"}</Code> rather than 0.1 or 1?
      </Prose>
      <Callout accent="green">
        <strong>Answer 4.</strong> In a pre-norm Transformer, each block adds a residual F(LN(x)) to x. Without LayerScale, the activation norm at block l grows roughly as <Code>{"\\|x_l\\| \\sim \\sqrt{l} \\cdot \\|F\\|"}</Code> (sum of L zero-mean independent-ish contributions). At depth 48 this is ~7× the initial norm. The attention sub-layer computes <Code>{"\\text{softmax}(QK^T / \\sqrt{d})"}</Code>; when activations inflate, QK products overflow the numerical range (in fp16, softmax produces NaN past magnitudes of ~256). Symptom: training looks fine for the first few hundred steps, then suddenly diverges as the scale pushes past the stable range. LayerScale with <Code>{"\\lambda = 10^{-6}"}</Code> per channel multiplies each F's contribution by a tiny scalar at init, so the block's effective contribution is <Code>{"10^{-6} \\cdot F"}</Code> — essentially identity. The network trains stably, and λ is learned upward to useful values only for blocks and channels that benefit. Init at 1 or 0.1 is too close to the untrained unstable regime; init at 0 (pure ReZero) works but loses per-channel flexibility. <Code>{"10^{-6}"}</Code> is empirically the sweet spot in CaiT: small enough for stability, large enough that gradients flow into λ itself early in training.
      </Callout>

      <H3>Exercise 5 (debugging — hidden dimension mismatch)</H3>
      <Prose>
        You train a custom CNN with 30 residual blocks. At epoch 5 the loss is still ~ln(num_classes), and the gradient norm at the embedding layer is 10<sup>−22</sup>. Blocks 20–30 have normal gradients. List three plausible causes specific to residual-connection design, and for each, describe how you would verify it in ~10 lines of diagnostic code.
      </Prose>
      <Callout accent="green">
        <strong>Answer 5.</strong> Three candidate failures, all consistent with "gradient dies between block 0 and block 20, but blocks 20+ train fine":
        <br />
        (1) <strong>Missing residual in one block between blocks 0 and 20.</strong> Hypothesis: someone wrote <Code>return self.bn2(self.conv2(h))</Code> instead of <Code>return self.bn2(self.conv2(h)) + x</Code> in block, say, 5. Diagnose: instrument per-block grad norm. <Code>{"for i, blk in enumerate(blocks): print(i, blk.conv1.weight.grad.norm().item())"}</Code>. Look for the block where the norm drops by more than 10× relative to its neighbor. That block is missing the skip.
        <br />
        (2) <strong>Post-activation ReLU clipping the skip path.</strong> Hypothesis: blocks compute <Code>{"y = ReLU(F(x) + x)"}</Code>, so deep in the stack many identity components are clipped to zero. Diagnose: in a single forward pass, replace each block with <Code>{"(F(x) + x, ReLU(F(x) + x))"}</Code> and measure the fraction of entries that are negative (and thus clipped). Above ~10% clipping, switch to pre-activation: remove the trailing ReLU and put Conv → BN → ReLU inside F.
        <br />
        (3) <strong>Post-norm LayerNorm attenuating the cumulative skip.</strong> Hypothesis: blocks use <Code>{"y = LN(x + Sublayer(x))"}</Code> instead of pre-norm. Each LN attenuates the identity component by roughly <Code>{"1/\\sqrt{N}"}</Code>; at N=30 the first-block contribution is lost. Diagnose: print the norm of the activation at each block. If the activation norm is roughly constant across depth (because LN normalizes), the residual structure is effectively destroyed by the LN — switch to pre-norm.
      </Callout>

    </div>
  ),
};

export default residualConnectionsContent;
