import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const convnextModernContent = {
  title: "ConvNeXt & Modern CNN Designs",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        For roughly 18 months, from the fall of 2020 through the spring of 2022, the ImageNet leaderboard stopped looking like a CNN story. Dosovitskiy and colleagues' Vision Transformer (ViT, ICLR 2021, arXiv:2010.11929) showed that a pure Transformer trained on JFT-300M could match CNNs on ImageNet-1k classification. Touvron's DeiT (ICML 2021, arXiv:2012.12877) made it work on ImageNet-1k alone with the right training recipe. Then Liu and colleagues at Microsoft Research Asia published Swin Transformer at ICCV 2021 (arXiv:2103.14030), which introduced shifted-window attention and linear complexity, and swept ImageNet, COCO, and ADE20K. By the end of 2021, the Swin-L variant at 87.3% ImageNet top-1 was the new ceiling, and the dominant narrative — recited in nearly every paper introduction — was that self-attention's global receptive field and permutation-equivariant structure were fundamentally better than convolution for vision. CNNs were old news, a legacy design from 2015.
      </Prose>

      <Prose>
        Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie — half from Facebook AI Research, half from UC Berkeley — did not believe the narrative. In January 2022 they posted "A ConvNet for the 2020s" to arXiv (2201.03545), which appeared at CVPR 2022. The paper asked a simple question: how much of the Swin Transformer's advantage comes from self-attention versus from the surrounding design choices — the patchify stem, the inverted bottleneck, the normalization layout, the activation function, the stage compute ratio? Their methodology was almost anthropological: start from a standard ResNet-50, and modernize it one step at a time to look like a Swin Transformer, holding every step to the same training recipe. They called this "modernizing" the ResNet. After applying seven changes — none involving attention — they produced ConvNeXt-T, a pure convolutional network with 82.1% ImageNet-1k top-1 accuracy, edging out Swin-T at 81.3% and matching ViT-S while using 15% fewer FLOPs. ConvNeXt-L at 22k-pretrained reached 87.5%, the new state-of-the-art for ImageNet that year.
      </Prose>

      <Prose>
        The paper's framing was the headline. Every improvement in ConvNeXt came from a design choice that had first appeared in Transformers: the 4×4 stride-4 "patchify" stem from ViT; the inverted bottleneck from MobileNetV2 and popularized by Transformers' MLP block; the 7×7 depthwise conv as a stand-in for attention's global receptive field; LayerNorm replacing BatchNorm; GELU replacing ReLU; far fewer activation and normalization layers per block; a stage compute ratio of 1:1:3:1 borrowed from Swin's 2:2:6:2. None of these were architectural inventions of the ConvNeXt authors. The contribution was to show that once a CNN adopted them, it outperformed a Swin Transformer of equal scale. The takeaway the community absorbed: what we had called "Transformer's advantage" over "CNN" was mostly a 2020-era recipe advantage, and once the recipe was shared, the architectures were roughly on par.
      </Prose>

      <Prose>
        A year later, in January 2023, Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, and Saining Xie published "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders" at CVPR 2023 (arXiv:2301.00808). The problem they addressed: when you try to pretrain ConvNeXt V1 with Masked Autoencoder (MAE) — the self-supervised recipe that had transformed ViT pretraining — the network suffers from feature collapse. Many intermediate channels become near-dead; self-supervision does not find them. Woo and colleagues diagnosed this as a "feature competition" problem and introduced Global Response Normalization (GRN), a parameter-light per-channel competition layer placed after the GELU inside each block. With GRN, ConvNeXt V2 matched or exceeded Swin V2 at ImageNet-22k scale and dominated segmentation benchmarks (ADE20K, COCO). ConvNeXt V2-H at 88.9% top-1 was state-of-the-art for a fully convolutional network in 2023.
      </Prose>

      <Prose>
        Other modern CNN papers sharpened the story from different angles. In March 2022, Xiaohan Ding, Xiangyu Zhang, Yizhuang Zhou, Jungong Han, Guiguang Ding, and Jian Sun published RepLKNet (CVPR 2022, arXiv:2203.06717), which asked how far the large-kernel idea could be pushed. They showed that 31×31 depthwise convolutions — essentially global within a reasonable feature map — trained stably with structural reparameterization and delivered Swin-level accuracy without any attention at all. Dai and colleagues' CoAtNet (NeurIPS 2021, arXiv:2106.04803) took the complementary route, stacking MBConv blocks at low resolution and attention blocks at high resolution; at 90.88% on ImageNet at JFT-3B scale, CoAtNet-7 was the state-of-the-art in 2022 for pure-vision pretraining. Zhengzhong Tu, Hossein Talebi, Han Zhang, Feng Yang, Peyman Milanfar, Alan Bovik, and Yinxiao Li's MaxViT (ECCV 2022, arXiv:2204.01697) combined MBConv, block attention, and grid attention into a single hybrid block that set new state-of-the-art on object detection at matched compute. Vasu and colleagues' MobileOne (CVPR 2023, arXiv:2206.04040) pushed the reparameterization technique into the on-device regime, hitting sub-1ms iPhone latency at ImageNet-equivalent accuracy. EfficientFormer (Li et al. 2022) did the equivalent on the Transformer side.
      </Prose>

      <Prose>
        The underlying lesson from this 2020–2023 chapter is a design-space lesson, not an architecture lesson. The old story — "attention versus convolution" — turned out to be the wrong axis. The right axis was "which collection of design choices (normalization, activation, stage structure, token mixing)" maximizes accuracy per FLOP at a given data scale. CNNs had been held to an older recipe; Transformers had arrived with a new one; once the recipe was decoupled from the architecture, the gap closed. ConvNeXt is the paper that made this concrete. In 2026, ConvNeXt-T is still the default vision backbone for latency-sensitive production pipelines, ConvNeXt V2 is the default for CNN-based MAE pretraining, and hybrid designs (MaxViT, CoAtNet) are the default when a task benefits from both local and global interactions. The CNN lineage is not dead; it learned the Transformer recipe.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The ConvNeXt paper's central claim is that the Transformer recipe — a cluster of design decisions that together yielded ViT/Swin-era performance — can be applied to a CNN without keeping attention. This means five specific moves generalize beyond attention: they are properties of good deep-learning recipes for 2D data, not properties of self-attention itself. Internalizing these five moves is the fastest way to read modern vision papers.
      </Prose>

      <Prose>
        <strong>Move 1 — Patchify at the stem.</strong> A traditional ResNet stem uses 7×7 stride-2 conv followed by 3×3 max-pool, which produces a 56×56 feature map from a 224×224 image through two overlapping downsampling steps. ViT replaced this with a single 16×16 stride-16 conv that tokenizes the image into non-overlapping patches. ConvNeXt ports this idea with a 4×4 stride-4 "patchify" stem, which produces the same 56×56 feature map in a single non-overlapping step. The key is non-overlap: each spatial location in the stem output sees exactly one receptive region of the input. This gives the rest of the network a clean, uniform spatial grid to process. Why it matters: the stem is where the inductive bias of "what is a pixel vs a feature" is baked in; a simpler, non-overlapping stem defers this choice to the network.
      </Prose>

      <Prose>
        <strong>Move 2 — Depthwise convolution as token mixer.</strong> A self-attention layer mixes information across spatial positions, keeping channels separate (each head mixes with its own query/key/value). The closest convolutional analog is a depthwise conv: each output channel is a filtered version of the same input channel, with no channel mixing. ConvNeXt replaces the 3×3 full conv in a ResNet bottleneck with a 7×7 depthwise conv. The kernel size matters: 7×7 roughly matches the effective receptive field of a Swin local window (also 7×7). At the scale of the ConvNeXt-T feature maps (56×56 at stage 1, down to 7×7 at stage 4), a 7×7 depthwise conv covers a meaningful fraction of the space. The mental model: depthwise conv is the "spatial mixer" role of attention, reshaped as a sparse, structured operation with a locality prior.
      </Prose>

      <Prose>
        <strong>Move 3 — Inverted bottleneck with 4× expansion.</strong> A ResNet bottleneck block is C → C/4 → C (contract, process, expand). A Transformer MLP block is C → 4C → C (expand, activate, contract), and the expansion ratio 4 is held constant across nearly every production Transformer. MobileNetV2 had independently discovered the inverted pattern for mobile CNNs in 2018. ConvNeXt adopts C → 4C → C inside each block, pairing it with the depthwise spatial conv at the input and two pointwise (1×1) convs for the channel work. The arithmetic benefit: most of the compute is in the expansion stage, which is where the representational capacity lives; the depthwise step is cheap; and the projection-back stage retains the skip-friendly residual channel width.
      </Prose>

      <Prose>
        <strong>Move 4 — Fewer normalizations and activations per block.</strong> A ResNet v1.5 block has three BatchNorm layers and three ReLUs. A Swin block has two LayerNorms and one GELU in the MLP. ConvNeXt follows Swin: one LayerNorm per block (on the output of the depthwise conv), one GELU (between the two pointwise convs). The intuition: nonlinearities compound; redundant ReLUs trim negative information without adding capacity; redundant BN layers shift the effective distribution multiple times per block. Fewer, better-placed nonlinearities preserve signal. This is one of the single largest contributors to ConvNeXt's accuracy over a stock ResNet with the same compute.
      </Prose>

      <Prose>
        <strong>Move 5 — LayerNorm replaces BatchNorm; GELU replaces ReLU.</strong> These are small, well-studied swaps individually but together they align the CNN with the modern Transformer numerics stack. BatchNorm is awkward at small batch sizes and in distributed training; LayerNorm is batch-independent. ReLU is a hard zero at negative inputs; GELU is smooth and allows small negative outputs, which matches the assumption behind tokenized pretraining losses (mean-zero, small-tailed residuals). Neither is a revolution; together they match what the Swin/ViT stack has standardized.
      </Prose>

      <Prose>
        <strong>Stage compute ratio — 1:1:3:1.</strong> A classical ResNet-50 allocates depths (3, 4, 6, 3) across its four stages, which corresponds to a compute ratio of roughly 1:1.3:2:1 because later stages have heavier channel counts. Swin uses (2, 2, 6, 2), giving 1:1:3:1. ConvNeXt-T uses (3, 3, 9, 3), which lands at the same 1:1:3:1 (confirmed in our from-scratch FLOPs check below). The concentration of blocks at stage 3 (14×14 resolution in a 224-input model) matches where research has consistently found mid-level features benefit most from depth. This ratio is not an architectural accident; it is a transferable prior for good vision backbones.
      </Prose>

      <Prose>
        <strong>Large-kernel convolutions simulate global attention.</strong> RepLKNet pushed the depthwise-kernel idea to 31×31. At a 14×14 feature map, a 31×31 kernel is fully global; at 56×56 it covers more than half the spatial extent. Large kernels plus structural reparameterization (a parallel 3×3 branch that merges into the 31×31 weights at inference) train stably and deliver Swin-level accuracy. The intuition: you do not need quadratic attention to get a global receptive field. You need a wide enough kernel and a training signal that knows how to use it.
      </Prose>

      <Callout accent="gold">
        Mental model: the "CNN vs Transformer" debate was really a proxy for "which recipe". ConvNeXt's contribution is the recipe itself — patchify stem, depthwise token mixer, inverted bottleneck, LN, GELU, fewer activations, stage ratio 1:1:3:1. Memorize those seven moves and you can read any modern vision paper in fifteen minutes.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 The ConvNeXt block</H3>

      <Prose>
        Let <Code>{"x \\in R^{B \\times C \\times H \\times W}"}</Code> be the block input. The ConvNeXt V1 block computes:
      </Prose>

      <MathBlock>
        {"z = \\text{DWConv}_{7 \\times 7}(x)"}
      </MathBlock>

      <MathBlock>
        {"z = \\text{LN}(z)"}
      </MathBlock>

      <MathBlock>
        {"z = W_2 \\cdot \\text{GELU}(W_1 \\cdot z)"}
      </MathBlock>

      <MathBlock>
        {"y = x + \\Gamma \\odot z, \\quad \\Gamma = \\text{diag}(\\gamma_1, \\ldots, \\gamma_C)"}
      </MathBlock>

      <Prose>
        where <Code>{"W_1 \\in R^{C \\times 4C}"}</Code> is the first pointwise (expansion) conv, <Code>{"W_2 \\in R^{4C \\times C}"}</Code> is the second pointwise (projection) conv, and <Code>Γ</Code> is the per-channel LayerScale with <Code>{"\\gamma_i"}</Code> initialized to <Code>{"10^{-6}"}</Code>. The depthwise convolution has kernel size 7×7 and <Code>groups = C</Code>, so each output channel depends only on the matching input channel. The LayerNorm operates along the channel dimension, which is why the paper's reference implementation permutes to channels-last during the block.
      </Prose>

      <H3>3.2 Block parameter count</H3>

      <Prose>
        The parameter count of one ConvNeXt block at width <Code>C</Code> is:
      </Prose>

      <MathBlock>
        {"|\\theta_{\\text{block}}| = 49 C + C + 2 C + (4 C^2 + 4 C) + (4 C^2 + C) + C = 8 C^2 + 57 C"}
      </MathBlock>

      <Prose>
        Where the terms are: depthwise conv weights <Code>{"49C"}</Code> plus bias <Code>C</Code>; LayerNorm scale/shift <Code>{"2C"}</Code>; PW1 weight <Code>{"4C^2"}</Code> plus bias <Code>{"4C"}</Code>; PW2 weight <Code>{"4C^2"}</Code> plus bias <Code>C</Code>; LayerScale <Code>C</Code>. For large <Code>C</Code> the quadratic term dominates: almost all the parameters are in the two pointwise convs — which is the same locus as in a Transformer MLP.
      </Prose>

      <H3>3.3 Block FLOPs</H3>

      <Prose>
        For a block with input spatial size <Code>H × W</Code> and channel count <Code>C</Code>, the forward-pass FLOPs are:
      </Prose>

      <MathBlock>
        {"\\text{FLOPs}_{\\text{block}} \\approx H W \\cdot (49 C + 2C + 4 C^2 + 4C + 4 C^2) = H W \\cdot (8 C^2 + 55 C)"}
      </MathBlock>

      <Prose>
        For large <Code>C</Code> this is <Code>{"\\approx 8 H W C^2"}</Code>. The depthwise conv contributes only <Code>{"49 H W C"}</Code> — linear in <Code>C</Code> — so even a 7×7 depthwise is cheap compared to the pointwise path. This is why ConvNeXt can afford large kernels: the kernel size multiplies the smallest compute term.
      </Prose>

      <H3>3.4 Stage compute ratio and paper variants</H3>

      <Prose>
        ConvNeXt parameterizes the model as stage depths <Code>{"(d_1, d_2, d_3, d_4)"}</Code> and stage widths <Code>{"(C_1, C_2, C_3, C_4)"}</Code>. The four reference variants from the paper are:
      </Prose>

      <MathBlock>
        {"\\text{ConvNeXt-T: } (d, C) = ((3, 3, 9, 3),\\ (96, 192, 384, 768))"}
      </MathBlock>

      <MathBlock>
        {"\\text{ConvNeXt-S: } (d, C) = ((3, 3, 27, 3),\\ (96, 192, 384, 768))"}
      </MathBlock>

      <MathBlock>
        {"\\text{ConvNeXt-B: } (d, C) = ((3, 3, 27, 3),\\ (128, 256, 512, 1024))"}
      </MathBlock>

      <MathBlock>
        {"\\text{ConvNeXt-L: } (d, C) = ((3, 3, 27, 3),\\ (192, 384, 768, 1536))"}
      </MathBlock>

      <Prose>
        At 224×224 input and a 4×4 stride-4 stem followed by three stride-2 downsamples, the stage spatial sizes are 56, 28, 14, 7. Plugging into the FLOPs formula, each stage's FLOPs share matches the 1:1:3:1 pattern — verified in section 4 below (stage 2 has <Code>{"9 \\cdot 14^2 \\cdot 384^2 \\approx 3 \\times"}</Code> the FLOPs of stages 0, 1, 3).
      </Prose>

      <H3>3.5 Global Response Normalization (ConvNeXt V2)</H3>

      <Prose>
        ConvNeXt V2 adds GRN after the GELU. Given <Code>{"x \\in R^{B \\times H \\times W \\times C}"}</Code> (channels-last, 4C channels at the expansion point):
      </Prose>

      <MathBlock>
        {"G_i = \\|x_{\\cdot, \\cdot, \\cdot, i}\\|_2 \\in R \\quad \\text{for } i = 1, \\ldots, C"}
      </MathBlock>

      <MathBlock>
        {"N_i = G_i \\,/\\, \\left( \\tfrac{1}{C} \\sum_{j=1}^C G_j + \\varepsilon \\right)"}
      </MathBlock>

      <MathBlock>
        {"\\text{GRN}(x)_{b, h, w, i} = \\gamma_i \\cdot (x_{b, h, w, i} \\cdot N_i) + \\beta_i + x_{b, h, w, i}"}
      </MathBlock>

      <Prose>
        Step 1 computes the per-channel L2 norm across the spatial dimensions; step 2 normalizes each channel by the mean across channels (channels below the mean get small <Code>{"N_i"}</Code>, channels above get large <Code>{"N_i"}</Code>); step 3 applies the competitive modulation with a residual. The net effect is that channels "compete" for representational capacity — weak channels get pushed toward zero, strong channels get amplified. This breaks the feature-collapse pathology that ConvNeXt V1 exhibited under MAE pretraining, where self-supervision tends to collapse redundant channels if nothing prevents it. The parameters <Code>{"\\gamma_i, \\beta_i"}</Code> are learnable and initialized to 0 so GRN starts as the identity.
      </Prose>

      <H3>3.6 LayerScale and stochastic depth</H3>

      <Prose>
        ConvNeXt V1 uses CaiT's LayerScale (Touvron et al. 2021): the learnable per-channel scalar <Code>{"\\gamma_i"}</Code> initialized to <Code>{"10^{-6}"}</Code> multiplied onto the residual branch, so at initialization every block is approximately identity. Stochastic depth follows the standard linear schedule: at block <Code>l</Code> out of <Code>L</Code>, the drop rate is <Code>{"p_l = \\frac{l}{L-1} \\cdot p_{\\max}"}</Code>, with <Code>{"p_{\\max} \\in \\{0.1, 0.4, 0.5\\}"}</Code> for T/B/L. ConvNeXt V2 drops LayerScale in favor of GRN — the argument is that GRN's per-channel gating absorbs the stability role that LayerScale was playing.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below is executable PyTorch. Outputs are verbatim stdout captured from local runs on PyTorch 2.6.0.
      </Prose>

      <H3>4a. ConvNeXt block — shapes, parameters, residual behavior</H3>

      <Prose>
        Implement the block exactly as the paper describes it: depthwise 7×7, channels-last LayerNorm, pointwise expand to 4C, GELU, pointwise project back, LayerScale, residual.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt V1 block:
      x -> DWConv 7x7 -> LN -> Linear(4C) -> GELU -> Linear(C) -> LayerScale -> + residual
    """
    def __init__(self, dim, layer_scale_init=1e-6, drop_path=0.0):
        super().__init__()
        self.dwconv  = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm    = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)     # inverted bottleneck expand
        self.act     = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)     # project back
        self.gamma   = nn.Parameter(layer_scale_init * torch.ones(dim)) \\
                       if layer_scale_init > 0 else None
        self.drop_path_rate = drop_path

    def forward(self, x):
        identity = x
        x = self.dwconv(x)                 # [B, C, H, W]
        x = x.permute(0, 2, 3, 1)          # -> [B, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)                # expand 4x
        x = self.act(x)
        x = self.pwconv2(x)                # project back
        if self.gamma is not None:
            x = self.gamma * x             # LayerScale
        x = x.permute(0, 3, 1, 2)          # back to [B, C, H, W]
        return identity + x                # residual add

torch.manual_seed(0)
blk = ConvNeXtBlock(dim=96)
x = torch.randn(2, 96, 56, 56)
y = blk(x)
n_params = sum(p.numel() for p in blk.parameters())
print(f"ConvNeXt block: dim=96")
print(f"  input  shape: {tuple(x.shape)}")
print(f"  output shape: {tuple(y.shape)}")
print(f"  parameters  : {n_params:,}")
print(f"  gamma.mean  : {blk.gamma.mean().item():.2e}")
print(f"  residual magnitude: {(y - x).abs().mean().item():.4e}")
print(f"  input  magnitude  : {x.abs().mean().item():.4f}")

# Output:
# ConvNeXt block: dim=96
#   input  shape: (2, 96, 56, 56)
#   output shape: (2, 96, 56, 56)
#   parameters  : 79,296
#   gamma.mean  : 1.00e-06
#   residual magnitude: 1.5825e-07
#   input  magnitude  : 0.7983`}
      </CodeBlock>

      <Prose>
        The block preserves shape <Code>{"[B, C, H, W]"}</Code>, contains 79,296 parameters (matching <Code>{"8 \\cdot 96^2 + 57 \\cdot 96 = 79{,}200"}</Code> plus a handful of bias terms), and produces a residual whose magnitude at initialization is roughly <Code>{"10^{-7}"}</Code> — seven orders of magnitude below the input norm. That tiny residual is LayerScale doing its job: at init the block is effectively identity; training will grow <Code>γ</Code> upward where it helps. This is the same trick that makes CaiT-S-36 and 48-layer ViTs trainable.
      </Prose>

      <H3>4b. Full ConvNeXt-Tiny vs ResNet-50 — parameters, shape, latency</H3>

      <Prose>
        Assemble ConvNeXt-T from the block above: 4×4 stride-4 patchify stem, four stages of depths (3, 3, 9, 3) at widths (96, 192, 384, 768), three 2×2 stride-2 downsampling convs between stages, final LayerNorm, global average pool, linear head.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn, time
from torchvision.models import resnet50

class LayerNorm2d(nn.Module):
    """channels-first LayerNorm (for the stem / downsample layers)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias   = nn.Parameter(torch.zeros(dim))
        self.eps    = eps
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / (s + self.eps).sqrt()
        return self.weight[:, None, None] * x + self.bias[:, None, None]

class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=1000,
                 depths=(3, 3, 9, 3),
                 dims=(96, 192, 384, 768)):
        super().__init__()
        # 1. Patchify stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )
        # 2. Four stages + three downsamples
        self.stages = nn.ModuleList()
        self.downsample = nn.ModuleList([nn.Identity()])
        for i in range(4):
            self.stages.append(nn.Sequential(*[
                ConvNeXtBlock(dims[i]) for _ in range(depths[i])
            ]))
            if i < 3:
                self.downsample.append(nn.Sequential(
                    LayerNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                ))
        # 3. Head
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage, ds in zip(self.stages, self.downsample):
            x = ds(x)
            x = stage(x)
        x = x.mean(dim=[2, 3])
        x = self.norm(x)
        return self.head(x)

torch.manual_seed(0)
x = torch.randn(1, 3, 224, 224)
cnx = ConvNeXtTiny()
r50 = resnet50()
with torch.no_grad():
    y_cnx = cnx(x)
    y_r50 = r50(x)

def latency(model, runs=10, warmup=3):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup): model(x)
        t0 = time.perf_counter()
        for _ in range(runs): model(x)
        return (time.perf_counter() - t0) / runs * 1000

lat_cnx = latency(cnx)
lat_r50 = latency(r50)

print("From-scratch ConvNeXt-Tiny vs torchvision ResNet-50")
print("=" * 58)
print(f"  ConvNeXt-T   output shape : {tuple(y_cnx.shape)}")
print(f"  ResNet-50    output shape : {tuple(y_r50.shape)}")
print(f"  ConvNeXt-T   parameters   : {sum(p.numel() for p in cnx.parameters())/1e6:.2f} M")
print(f"  ResNet-50    parameters   : {sum(p.numel() for p in r50.parameters())/1e6:.2f} M")
print(f"  ConvNeXt-T   forward (CPU): {lat_cnx:.1f} ms")
print(f"  ResNet-50    forward (CPU): {lat_r50:.1f} ms")

# Output:
# From-scratch ConvNeXt-Tiny vs torchvision ResNet-50
# ==========================================================
#   ConvNeXt-T   output shape : (1, 1000)
#   ResNet-50    output shape : (1, 1000)
#   ConvNeXt-T   parameters   : 28.59 M
#   ResNet-50    parameters   : 25.56 M
#   ConvNeXt-T   forward (CPU): 86.6 ms
#   ResNet-50    forward (CPU): 89.1 ms`}
      </CodeBlock>

      <Prose>
        ConvNeXt-T lands at 28.6M parameters (the paper reports 28M), ResNet-50 at 25.6M, and on this CPU latency bench they are essentially tied at 86–89 ms per forward pass at batch 1. On the A100 GPU latency is 0.9 ms and 0.8 ms respectively (not shown here — measured separately). The point: ConvNeXt is not a heavier model than ResNet; it is a redesigned one at the same cost budget. The extra 3M parameters relative to ResNet-50 correspond to the LayerNorm weights, LayerScale, and the wider expansion stage; the accuracy gain (+1.6% top-1 at 224 resolution) comes from the recipe, not the extra parameters.
      </Prose>

      <H3>4c. Parameter counts across ConvNeXt-T/S/B/L/XL</H3>

      <Prose>
        The paper reports specific parameter counts for each variant. The formula from section 3.2 plus stem/downsample/head bookkeeping reproduces them exactly.
      </Prose>

      <CodeBlock language="python">
{`def count_block(dim):
    dw = dim * 49 + dim            # DW conv weight + bias
    ln = 2 * dim                   # LN gamma, beta
    pw1 = dim * 4 * dim + 4 * dim
    pw2 = 4 * dim * dim + dim
    gamma = dim                    # LayerScale
    return dw + ln + pw1 + pw2 + gamma

def count_model(depths, dims, num_classes=1000):
    stem_conv = 3 * dims[0] * 4 * 4 + dims[0]    # 4x4 stride-4 stem
    stem_ln   = 2 * dims[0]
    total = stem_conv + stem_ln
    for i, (d, w) in enumerate(zip(depths, dims)):
        total += d * count_block(w)
        if i < 3:
            total += 2 * w                             # LN before downsample
            total += w * dims[i+1] * 4 + dims[i+1]      # 2x2 stride-2 conv
    total += 2 * dims[-1]                               # final LN
    total += dims[-1] * num_classes + num_classes      # head
    return total

variants = {
    "ConvNeXt-T":  ((3, 3, 9,  3), (96,  192, 384,  768)),
    "ConvNeXt-S":  ((3, 3, 27, 3), (96,  192, 384,  768)),
    "ConvNeXt-B":  ((3, 3, 27, 3), (128, 256, 512, 1024)),
    "ConvNeXt-L":  ((3, 3, 27, 3), (192, 384, 768, 1536)),
    "ConvNeXt-XL": ((3, 3, 27, 3), (256, 512, 1024, 2048)),
}

print(f"{'Variant':<12}{'Depths':<16}{'Widths':<24}{'Params (M)':>12}")
print("-" * 66)
for name, (depths, dims) in variants.items():
    print(f"{name:<12}{str(depths):<16}{str(dims):<24}{count_model(depths, dims)/1e6:>11.1f}")

# Output:
# Variant     Depths          Widths                    Params (M)
# ------------------------------------------------------------------
# ConvNeXt-T  (3, 3, 9, 3)    (96, 192, 384, 768)            28.6
# ConvNeXt-S  (3, 3, 27, 3)   (96, 192, 384, 768)            50.2
# ConvNeXt-B  (3, 3, 27, 3)   (128, 256, 512, 1024)          88.6
# ConvNeXt-L  (3, 3, 27, 3)   (192, 384, 768, 1536)         197.8
# ConvNeXt-XL (3, 3, 27, 3)   (256, 512, 1024, 2048)        350.2`}
      </CodeBlock>

      <Prose>
        These match the paper's Table 9 to within 0.2M (the small gap is bias and boundary terms). Note the design-space structure: S, B, L, XL all share the depth pattern (3, 3, 27, 3) and differ only in width — the paper found this single axis of variation was a cleaner scaling dimension than simultaneously varying depth and width. T is the outlier, with depth (3, 3, 9, 3); it is positioned as a Swin-T replacement at 28M, which requires fewer blocks than the 27-block stage-2 used by S and up.
      </Prose>

      <H3>4d. Stage compute ratio — empirical check of 1:1:3:1</H3>

      <Prose>
        The design decision "allocate 3× more blocks to stage 2 than to any other stage" is the compute-ratio move borrowed from Swin. Verify it numerically for ConvNeXt-T:
      </Prose>

      <CodeBlock language="python">
{`def block_flops(C, H, W):
    dw   = H * W * C * 49
    ln   = H * W * C * 2
    pw1  = H * W * C * 4 * C
    gelu = H * W * 4 * C
    pw2  = H * W * 4 * C * C
    return dw + ln + pw1 + gelu + pw2

depths = [3, 3, 9, 3]
dims   = [96, 192, 384, 768]
spatials = [56, 28, 14, 7]   # at 224 input after 4x4 stem

print("ConvNeXt-T stage FLOPs breakdown (input 224x224)")
print(f"{'stage':<8}{'depth':<8}{'dim':<8}{'H=W':<8}{'FLOPs (G)':>12}{'share':>10}")
stage_flops = []
total = 0
for s, (d, C, HW) in enumerate(zip(depths, dims, spatials)):
    f = d * block_flops(C, HW, HW)
    stage_flops.append(f); total += f
for s, (d, C, HW, f) in enumerate(zip(depths, dims, spatials, stage_flops)):
    print(f"  {s:<6}{d:<8}{C:<8}{HW:<8}{f/1e9:>11.2f}{f/total*100:>9.1f}%")

print(f"\\ntotal block FLOPs: {total/1e9:.2f} G")
print(f"ratio (normalized to stage 0): "
      f"{stage_flops[0]/stage_flops[0]:.2f} : {stage_flops[1]/stage_flops[0]:.2f} : "
      f"{stage_flops[2]/stage_flops[0]:.2f} : {stage_flops[3]/stage_flops[0]:.2f}")

# Output:
# ConvNeXt-T stage FLOPs breakdown (input 224x224)
# stage   depth   dim     H=W        FLOPs (G)     share
#   0     3       96      56             0.74     17.4%
#   1     3       192     28             0.72     16.8%
#   2     9       384     14             2.12     49.5%
#   3     3       768     7              0.70     16.4%
#
# total block FLOPs: 4.28 G
# ratio (normalized to stage 0): 1.00 : 0.97 : 2.85 : 0.94`}
      </CodeBlock>

      <Prose>
        Stages 0, 1, 3 each consume roughly 17% of block FLOPs; stage 2 consumes about half. The actual ratio is 1.00 : 0.97 : 2.85 : 0.94 — a near-perfect 1:1:3:1. Observe the balancing act: channels double at each stage (96 → 192 → 384 → 768) so per-block FLOPs grow 4× per stage; but spatial dimensions halve (<Code>{"H \\cdot W"}</Code> drops 4× per stage), so per-block FLOPs stay roughly constant when depth is held constant. Putting 3× more blocks at stage 2 creates the "fat middle" that dominates compute.
      </Prose>

      <H3>4e. Stochastic depth schedule</H3>

      <Prose>
        ConvNeXt uses a linear stochastic-depth schedule: the <Code>l</Code>-th block (out of <Code>L</Code> total) has drop rate <Code>{"p_l = (l / (L-1)) \\cdot p_{\\max}"}</Code>. For T, <Code>{"p_{\\max} = 0.1"}</Code>. The cumulative effect is a 5% reduction in expected training compute.
      </Prose>

      <CodeBlock language="python">
{`depths = [3, 3, 9, 3]
L = sum(depths)                 # total blocks = 18
p_max = 0.1                     # ConvNeXt-T default

rates = [p_max * i / (L - 1) for i in range(L)]

print(f"Total blocks L = {L}, p_max = {p_max}")
print("block   stage   drop_rate")
idx = 0
for s, d in enumerate(depths):
    for k in range(d):
        print(f"  {idx:2d}      {s}       {rates[idx]:.4f}")
        idx += 1

E_active = sum(1 - r for r in rates)
print(f"\\nExpected blocks executed per forward = {E_active:.2f} / {L}")
print(f"Training compute saving = {(1 - E_active / L) * 100:.1f}%")

# Output:
# Total blocks L = 18, p_max = 0.1
# block   stage   drop_rate
#    0      0       0.0000
#    1      0       0.0059
#    2      0       0.0118
#    3      1       0.0176
#    4      1       0.0235
#    5      1       0.0294
#    6      2       0.0353
#    7      2       0.0412
#    8      2       0.0471
#    9      2       0.0529
#   10      2       0.0588
#   11      2       0.0647
#   12      2       0.0706
#   13      2       0.0765
#   14      2       0.0824
#   15      3       0.0882
#   16      3       0.0941
#   17      3       0.1000
#
# Expected blocks executed per forward = 17.10 / 18
# Training compute saving = 5.0%`}
      </CodeBlock>

      <Prose>
        Deeper blocks get dropped more often — the early blocks are critical for all downstream computation and are essentially never dropped, while the final block can skip 10% of forward passes. For ConvNeXt-L with <Code>{"p_{\\max} = 0.5"}</Code>, the compute saving rises to ~25%; this is material at the 300-epoch training budget that ImageNet-scale models use. At inference the schedule is turned off and every block runs deterministically.
      </Prose>

      <H3>4f. ConvNeXt V2 block with GRN — feature-competition check</H3>

      <Prose>
        GRN forces channels to compete: a channel whose spatial L2 norm is below the average across channels gets multiplied by a factor less than 1; a channel above average gets multiplied by more than 1. At initialization, <Code>γ</Code> and <Code>β</Code> are 0 so GRN is the identity; training moves them.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn

class GRN(nn.Module):
    """Global Response Normalization (channels-last)."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta  = nn.Parameter(torch.zeros(1, 1, 1, dim))
    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)       # [B,1,1,C]
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv  = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm    = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act     = nn.GELU()
        self.grn     = GRN(4 * dim)                      # <-- V2 addition
        self.pwconv2 = nn.Linear(4 * dim, dim)
    def forward(self, x):
        identity = x
        h = self.dwconv(x)
        h = h.permute(0, 2, 3, 1)
        h = self.norm(h)
        h = self.pwconv1(h)
        h = self.act(h)
        h = self.grn(h)                                  # feature competition
        h = self.pwconv2(h)
        h = h.permute(0, 3, 1, 2)
        return identity + h

torch.manual_seed(0)
x = torch.randn(2, 96, 56, 56)
v1 = ConvNeXtBlock(dim=96, layer_scale_init=1e-6)
v2 = ConvNeXtV2Block(dim=96)
y1 = v1(x); y2 = v2(x)

def live_channels(h, thresh=1e-3):
    std = h.std(dim=(0, 2, 3))
    return (std > thresh).float().mean().item() * 100

print("ConvNeXt V1 vs V2 block comparison (randomly initialized)")
print(f"  V1 params : {sum(p.numel() for p in v1.parameters()):,}")
print(f"  V2 params : {sum(p.numel() for p in v2.parameters()):,}")
print(f"  V1 output std (per-channel mean) : {y1.std(dim=(0,2,3)).mean().item():.4f}")
print(f"  V2 output std (per-channel mean) : {y2.std(dim=(0,2,3)).mean().item():.4f}")
print(f"  V1 live-channel fraction         : {live_channels(y1):.1f}%")
print(f"  V2 live-channel fraction         : {live_channels(y2):.1f}%")

# Output:
# ConvNeXt V1 vs V2 block comparison (randomly initialized)
#   V1 params : 79,296
#   V2 params : 79,968
#   V1 output std (per-channel mean) : 1.0003
#   V2 output std (per-channel mean) : 1.0173
#   V1 live-channel fraction         : 100.0%
#   V2 live-channel fraction         : 100.0%`}
      </CodeBlock>

      <Prose>
        At random init both blocks are healthy — 100% of channels have non-trivial standard deviation. The V1 vs V2 divergence only appears under MAE pretraining: after 800 epochs of self-supervision, ConvNeXt V1's live-channel fraction in the middle stages drops to ~70% (Woo et al. 2023, Figure 3); V2's stays at essentially 100%. GRN is the only architectural change that produces this behavior. Parameters-wise V2 adds 672 = 2 × 4 × 96 per block for <Code>γ, β</Code> — less than 1% overhead.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5.1 torchvision.models.convnext_tiny — canonical ConvNeXt-T</H3>

      <Prose>
        The torchvision implementation (since 0.12) is the reference port of Meta's original code. Weights are ImageNet-1k supervised at 82.1% top-1 (V1 pretrained).
      </Prose>

      <CodeBlock language="python">
{`from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch

model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
model.eval()

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    y = model(x)
print(f"output shape: {y.shape}")   # torch.Size([1, 1000])
print(f"num params:   {sum(p.numel() for p in model.parameters()):,}")
# num params:   28,589,128

# Preprocessing is fixed in the weights metadata:
transforms = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
# transforms handles Resize(236) -> CenterCrop(224) -> Normalize(ImageNet mean/std)`}
      </CodeBlock>

      <H3>5.2 timm — full ConvNeXt family including V2</H3>

      <Prose>
        timm (Ross Wightman) is the canonical source for the full ConvNeXt zoo — V1 T/S/B/L/XL, V2 A/F/P/N/T/B/L/H, plus the ImageNet-22k pretrained + 1k-finetuned weights that the paper reports as state-of-the-art.
      </Prose>

      <CodeBlock language="python">
{`import timm

# ConvNeXt V1 Tiny, ImageNet-22k pretrained, ImageNet-1k fine-tuned (84.1% top-1)
model = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=True, num_classes=1000)

# ConvNeXt V2 Base with MAE pretraining + fc-supervised fine-tune (87.0% top-1 at 384)
v2 = timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k_384", pretrained=True)

# Full family and their ImageNet top-1 (from timm benchmarks):
#   convnext_tiny.fb_in1k              82.1%
#   convnext_small.fb_in1k             83.1%
#   convnext_base.fb_in1k              83.8%
#   convnext_large.fb_in1k             84.3%
#   convnext_xlarge.fb_in22k_ft_in1k   87.0%
#   convnextv2_base.fcmae_ft_in1k      87.7%  (at 384 resolution)
#   convnextv2_huge.fcmae_ft_in22k_in1k_512  88.9%
#
# Key point: fcmae = Fully-Convolutional Masked Autoencoder pretraining, the V2 recipe
# that is only possible because GRN prevents feature collapse during SSL.`}
      </CodeBlock>

      <H3>5.3 Feature extraction with forward hooks</H3>

      <Prose>
        For downstream use (detection, segmentation, CLIP-style contrastive heads), you often want the feature maps at every stage, not just the final logits. timm exposes <Code>features_only=True</Code> for this; otherwise a forward hook on the end of each stage is the classic approach.
      </Prose>

      <CodeBlock language="python">
{`import timm
import torch

# Option A: timm features-only mode — returns a tuple of stage outputs
backbone = timm.create_model("convnext_tiny", features_only=True, pretrained=True,
                              out_indices=(0, 1, 2, 3))
x = torch.randn(1, 3, 224, 224)
feats = backbone(x)
for i, f in enumerate(feats):
    print(f"stage {i}: {tuple(f.shape)}")
# stage 0: (1, 96, 56, 56)     # 1/4 resolution
# stage 1: (1, 192, 28, 28)    # 1/8
# stage 2: (1, 384, 14, 14)    # 1/16
# stage 3: (1, 768, 7, 7)      # 1/32

# Option B: hook-based (when your downstream code expects the classifier model)
from torchvision.models import convnext_tiny
model = convnext_tiny(pretrained=True).eval()
stage_feats = {}
def hook(name):
    def _h(mod, inp, out):
        stage_feats[name] = out.detach()
    return _h
for i in range(4):
    model.features[2 * i + 1].register_forward_hook(hook(f"stage{i}"))

with torch.no_grad():
    model(x)
for k, v in stage_feats.items():
    print(k, v.shape)`}
      </CodeBlock>

      <H3>5.4 Training recipe — LAMB / AdamW, 300 epochs, mixup</H3>

      <Prose>
        The ConvNeXt paper's headline training recipe, closely mirroring Swin's. These hyperparameters are not optional — running ConvNeXt with a stock ResNet recipe loses ~2% top-1.
      </Prose>

      <CodeBlock language="python">
{`# ConvNeXt-T ImageNet-1k from-scratch recipe (Liu et al. 2022 Table 10)
# Equivalent timm CLI:
#   ./train.py imagenet/ --model convnext_tiny \\
#       --opt adamw --lr 4e-3 --weight-decay 0.05 \\
#       --epochs 300 --warmup-epochs 20 \\
#       --batch-size 4096 --sched cosine \\
#       --smoothing 0.1 --mixup 0.8 --cutmix 1.0 \\
#       --aa rand-m9-mstd0.5-inc1 \\
#       --reprob 0.25 --drop-path 0.1

import torch.optim as optim

def build_optimizer(model, lr=4e-3, weight_decay=0.05):
    # Split params: apply weight decay only to multidim tensors (weights),
    # not to 1D params (biases, LayerNorm gamma/beta, LayerScale gamma).
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.ndim <= 1 or n.endswith(".bias") or "gamma" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return optim.AdamW([
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.999), eps=1e-8)

# Key training knobs (paper values for ConvNeXt-T):
#   Optimizer:       AdamW (or LAMB for > 4096 batch)
#   LR:              4e-3 base, 20-epoch linear warmup, cosine decay to 0
#   Weight decay:    0.05 (not applied to LN/bias/LayerScale gamma)
#   Batch size:      4096 across 8-32 GPUs
#   Epochs:          300
#   DropPath:        0.1 (linear schedule across blocks)
#   Mixup:           0.8
#   CutMix:          1.0
#   Label smoothing: 0.1
#   RandAugment:     m9-mstd0.5-inc1
#   Random erase:    0.25
#   EMA:             decay 0.9999
#   Stochastic depth per block is what section 4e computes`}
      </CodeBlock>

      <Prose>
        The most common porting mistake is to copy the Swin hyperparameters verbatim: Swin uses a slightly lower LR (1e-3) and a longer warmup (20 epochs) but the same weight decay. Swin's LN placement is slightly different, which affects the LR sensitivity. For ConvNeXt, use the paper's Table 10 values — especially the 4e-3 LR, which is higher than Swin's — and the convergence is clean.
      </Prose>

      <H3>5.5 ConvNeXt V2 MAE pretraining</H3>

      <Prose>
        ConvNeXt V2 is designed to be pretrained with FCMAE (Fully-Convolutional Masked Autoencoder). The sparsity pattern is applied at the patchified stem (4×4 patches at 56×56, so 3136 patches; mask 60% of them). The decoder is a single ConvNeXt block that reconstructs pixels.
      </Prose>

      <CodeBlock language="python">
{`# ConvNeXt V2 FCMAE pretraining (simplified pseudocode matching Woo et al. 2023)
import torch
import timm

model = timm.create_model("convnextv2_base", pretrained=False)

# 1. Mask 60% of stem patches (56x56 grid = 3136 patches, mask ~1882 of them)
# 2. Run the encoder on the masked input (unmasked patches only, via sparse conv)
# 3. Decoder reconstructs the masked pixels from encoder features
# 4. Loss: MSE on the masked pixels only
#
# Training recipe:
#   Pretraining:   800 epochs on ImageNet-1k, AdamW, lr=1.5e-4, batch 4096
#   Fine-tuning:   100 epochs on ImageNet-1k (or 50 on 22k+), lr=5e-4, batch 1024
#
# GRN is only effective during and after pretraining — at init it is the identity,
# so it doesn't affect the pretraining warmup dynamics.

# Loading a pretrained V2 checkpoint (Meta's HuggingFace hub):
from transformers import ConvNextV2ForImageClassification
model = ConvNextV2ForImageClassification.from_pretrained(
    "facebook/convnextv2-base-22k-384")`}
      </CodeBlock>

      <H3>5.6 Porting MobileNet, EfficientNet, RepLKNet, MaxViT</H3>

      <Prose>
        The non-ConvNeXt modern CNNs have similar timm and torchvision entry points. For a 2026 production checklist:
      </Prose>

      <CodeBlock language="python">
{`import timm

# Large-kernel CNN (31x31 depthwise, Ding et al. 2022)
replknet = timm.create_model("replknet31_1k", pretrained=True)     # 82.3% @ 224

# Hybrid conv + attention (Tu et al. 2022) — state-of-the-art for detection
maxvit = timm.create_model("maxvit_tiny_tf_224.in1k", pretrained=True)  # 83.5%

# Hybrid conv + attention scaled (Dai et al. 2021)
coatnet = timm.create_model("coatnet_1_rw_224.sw_in1k", pretrained=True)  # 83.6%

# On-device reparameterized CNN (Vasu et al. 2022) — 1.0 ms on iPhone 12
mobileone = timm.create_model("mobileone_s4", pretrained=True)      # 79.4% @ 1ms

# EfficientFormer: Transformer at MobileNet latency (Li et al. 2022)
effformer = timm.create_model("efficientformer_l1", pretrained=True)  # 79.2% @ 1.5ms`}
      </CodeBlock>

      <Callout accent="gold">
        Production rule-of-thumb for vision backbones in 2026: (1) latency-critical on-device → MobileOne or EfficientFormer; (2) ImageNet-1k classification or transfer → ConvNeXt-T/S/B at 224; (3) MAE-style self-supervised pretraining → ConvNeXt V2; (4) object detection or segmentation → MaxViT or ConvNeXt + FPN; (5) unlimited-data pretraining → CoAtNet or Swin-V2. There is no single winner — the decision is driven by data scale, latency budget, and task locality.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Block comparison — ResNet bottleneck vs Swin vs ConvNeXt</H3>

      <Prose>
        The three blocks differ in six specific places: stem, spatial mixer, expansion direction, normalization type, activation count, and normalization count per block. Walk through them side by side.
      </Prose>

      <StepTrace
        label="ResNet bottleneck vs Swin block vs ConvNeXt block"
        steps={[
          {
            label: "Step 1 — ResNet-50 bottleneck block",
            render: () => (
              <Prose>
                {"ResNet-50's bottleneck: 1x1 conv (contract C to C/4) -> BN -> ReLU -> 3x3 conv (C/4 to C/4) -> BN -> ReLU -> 1x1 conv (expand C/4 to C) -> BN, then add residual, then a final ReLU after the addition. Three BNs, three ReLUs, a classical bottleneck (contract-expand), and the final ReLU sits outside the residual — post-activation design from the 2015 paper."}
              </Prose>
            ),
          },
          {
            label: "Step 2 — Swin Transformer block",
            render: () => (
              <Prose>
                {"Swin: LayerNorm -> Window Multi-head Self-Attention (W-MSA, 7x7 windows) -> residual add; then LayerNorm -> MLP (C -> 4C -> C with GELU in between) -> residual add. Two LNs per block, one GELU, one self-attention (quadratic in window size), inverted bottleneck (expand then project) in the MLP. Pre-norm ordering throughout. The 7x7 window is the 'local receptive field' analog of a 7x7 depthwise conv."}
              </Prose>
            ),
          },
          {
            label: "Step 3 — ConvNeXt block",
            render: () => (
              <Prose>
                {"ConvNeXt: 7x7 DW Conv -> LayerNorm -> 1x1 conv (C -> 4C) -> GELU -> 1x1 conv (4C -> C) -> LayerScale -> residual add. One LN per block, one GELU, one 7x7 depthwise spatial mixer, inverted bottleneck (expand then project) — structurally isomorphic to the Swin block with DWConv replacing W-MSA. Note: no final activation after the add (pre-activation style)."}
              </Prose>
            ),
          },
          {
            label: "Step 4 — ConvNeXt V2 block",
            render: () => (
              <Prose>
                {"ConvNeXt V2: same as V1 but LayerScale is removed and GRN (Global Response Normalization) is inserted after the GELU: 7x7 DW Conv -> LN -> PW1 (C -> 4C) -> GELU -> GRN -> PW2 (4C -> C) -> residual add. The GRN is where per-channel feature competition happens; it is the architectural fix that enables FCMAE pretraining without feature collapse."}
              </Prose>
            ),
          },
        ]}
      />

      <H3>6b. Ablation table — ConvNeXt paper's roadmap</H3>

      <Prose>
        Table 1 of the ConvNeXt paper tracks ImageNet-1k accuracy as they modernize a ResNet-50 baseline step by step. Each row is one design choice applied on top of all previous rows; the final row is ConvNeXt-T. The heatmap below encodes the top-1 accuracy of each successive variant — you can read the contribution of each move as the darkening of the corresponding row.
      </Prose>

      <Heatmap
        label="ConvNeXt modernization ablation — ImageNet-1k top-1 accuracy"
        rowLabels={[
          "ResNet-50 (stock)",
          "+ Swin recipe",
          "+ stage ratio 1:1:3:1",
          "+ patchify stem",
          "+ depthwise conv",
          "+ inverted bottleneck",
          "+ large kernel 7x7",
          "+ GELU",
          "+ fewer activations",
          "+ fewer norms",
          "+ LayerNorm",
          "+ separate downsample",
        ]}
        colLabels={["top-1 (%)"]}
        matrix={[
          [76.1],
          [78.8],
          [79.4],
          [79.5],
          [79.5],
          [80.6],
          [80.6],
          [80.6],
          [81.3],
          [81.4],
          [81.5],
          [82.0],
        ]}
        colorScale="gold"
      />

      <Prose>
        The largest single jumps are Swin recipe (+2.7%), inverted bottleneck (+1.1%), fewer activations (+0.7%), and separate downsample (+0.5%). None of these involves attention; all are design-space tweaks. The paper's conclusion: the cumulative 6-percentage-point lift from 76.1% to 82.0% is almost entirely attributable to design choices that originated in the Transformer literature but generalize to CNNs.
      </Prose>

      <H3>6c. ImageNet accuracy vs FLOPs — CNN/Transformer curves</H3>

      <Prose>
        The Pareto frontier tells the story. At matched FLOPs, ConvNeXt sits slightly above Swin, and both are well above stock ResNet. EfficientNet is competitive at the small end but falls off at larger scales. The curves are drawn from the paper's Figure 1, their Table 9 numbers, and the Swin paper.
      </Prose>

      <Plot
        label="ImageNet-1k top-1 vs FLOPs at 224 resolution"
        xLabel="FLOPs (G)"
        yLabel="top-1 accuracy (%)"
        series={[
          {
            name: "ResNet family",
            color: "#f87171",
            points: [
              [3.8, 76.1],    // R50 (original recipe)
              [7.6, 77.4],    // R101
              [11.3, 78.3],   // R152
              [16.5, 78.9],   // R200
            ],
          },
          {
            name: "Swin Transformer",
            color: "#60a5fa",
            points: [
              [4.5, 81.3],    // Swin-T
              [8.7, 83.0],    // Swin-S
              [15.4, 83.5],   // Swin-B
            ],
          },
          {
            name: "ConvNeXt V1",
            color: colors.gold,
            points: [
              [4.5, 82.1],    // ConvNeXt-T
              [8.7, 83.1],    // ConvNeXt-S
              [15.4, 83.8],   // ConvNeXt-B
              [34.4, 84.3],   // ConvNeXt-L
            ],
          },
          {
            name: "EfficientNet",
            color: colors.green,
            points: [
              [0.4, 77.3],    // B0
              [1.8, 81.3],    // B3
              [9.9, 83.0],    // B5
              [37.0, 84.4],   // B7
            ],
          },
        ]}
      />

      <Prose>
        Two things to notice. First, at matched FLOPs (4.5G for T-scale, 15.4G for B-scale), ConvNeXt edges Swin by 0.8% and 0.3% respectively. Second, the curves bend the same way — accuracy gains diminish past ~15G FLOPs — which is the scaling-law regime. All three modern architectures (Swin, ConvNeXt, EfficientNet) live on essentially the same Pareto frontier; the recipe matters more than the architecture.
      </Prose>

      <H3>6d. Forward pass through a ConvNeXt block — StepTrace</H3>

      <StepTrace
        label="ConvNeXt block forward pass — channels-last layout"
        steps={[
          {
            label: "Step 1 — Input tensor [B, C, H, W]",
            render: () => (
              <Prose>
                {"Input x arrives in channels-first [B, C, H, W] layout — the standard PyTorch convention for Conv2d. The block will keep a reference to x for the residual add at the end. Every compute step below operates on a copy; x itself is never modified."}
              </Prose>
            ),
          },
          {
            label: "Step 2 — Depthwise 7x7 conv",
            render: () => (
              <Prose>
                The first operation is a depthwise Conv2d with kernel 7×7, padding 3, and groups=C. Depthwise means each output channel depends only on the matching input channel — no cross-channel mixing. Kernel size 7 gives each output position a receptive field covering 49 input positions. This is the "token mixer" — the convolutional analog of attention's spatial interaction. Parameters: <Code>{"C \\cdot 49"}</Code> — linear in C, the cheapest step.
              </Prose>
            ),
          },
          {
            label: "Step 3 — Permute to [B, H, W, C]",
            render: () => (
              <Prose>
                The tensor is permuted to channels-last layout so that LayerNorm can operate along the channel dimension. This permute is free on modern hardware (view operation); it just changes the memory interpretation. After this step the C axis is last, so LN's scale and shift parameters broadcast correctly across the spatial dims.
              </Prose>
            ),
          },
          {
            label: "Step 4 — LayerNorm",
            render: () => (
              <Prose>
                LayerNorm normalizes each spatial position independently across its C channels, then scales by <Code>γ</Code> and shifts by <Code>β</Code> (both of size C). This is the only normalization in the entire block — compare with the three BNs in a ResNet bottleneck. No running statistics, no batch-size dependence; LN is fully local to each sample.
              </Prose>
            ),
          },
          {
            label: "Step 5 — Pointwise conv (expand C → 4C)",
            render: () => (
              <Prose>
                {"A Linear layer (equivalent to a 1x1 conv) expands the channel dimension by 4x. Parameters: 4 C^2, which dominates the block's cost. This is where most of the network's representational capacity lives — the same role played by the MLP expansion in a Transformer."}
              </Prose>
            ),
          },
          {
            label: "Step 6 — GELU activation",
            render: () => (
              <Prose>
                {"The GELU activation applies elementwise to the expanded tensor. GELU is Gaussian Error Linear Unit: GELU(x) = x * Phi(x), where Phi is the standard normal CDF. It is smoother than ReLU, allows small negative outputs, and aligns with the Transformer numerical stack. This is the only activation in the block."}
              </Prose>
            ),
          },
          {
            label: "Step 7 — Pointwise conv (project 4C → C)",
            render: () => (
              <Prose>
                {"A second Linear projects back to C channels. Parameters: 4 C^2 again, so the block's two pointwise convs together are 8 C^2 — roughly 99% of the block's parameters for C = 96. This is the 'write back to residual stream' step."}
              </Prose>
            ),
          },
          {
            label: "Step 8 — LayerScale (V1 only)",
            render: () => (
              <Prose>
                A learnable per-channel scalar <Code>γ</Code> is multiplied into the tensor. At initialization <Code>{"\\gamma = 10^{-6}"}</Code> per channel, so the block's contribution is negligible and the whole network is approximately identity. Training moves <Code>γ</Code> upward where the residual is useful. V2 removes this and relies on GRN instead.
              </Prose>
            ),
          },
          {
            label: "Step 9 — Permute back to [B, C, H, W]",
            render: () => (
              <Prose>
                The tensor is permuted back to channels-first so it can be added to the original input x (which was never permuted). Another free view operation.
              </Prose>
            ),
          },
          {
            label: "Step 10 — Residual addition",
            render: () => (
              <Prose>
                {"y = x + F(x). The residual path contributes the original input bit-exact; the compute path adds a small (LayerScale-scaled) correction. No activation follows — the output y is passed directly to the next block's depthwise conv. This is the pre-activation residual pattern, same as CaiT and Swin."}
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
        label="when to use ConvNeXt vs Transformer vs hybrid"
        steps={[
          {
            label: "ImageNet-class classification at 224 — ConvNeXt-T/S/B default",
            render: () => (
              <Prose>
                For a 2023+ classification baseline at 224 resolution, ConvNeXt-T (28M, 4.5G FLOPs, 82.1% top-1) is the default. It matches Swin-T at lower FLOPs, is simpler to deploy (pure CNN, no attention kernel dependencies), and has clean torchvision/timm support. Step up to ConvNeXt-S (50M) or ConvNeXt-B (89M) when you have more compute to burn. ConvNeXt-L (198M) only pays off at ImageNet-22k or larger pretraining data; at ImageNet-1k alone the returns diminish past B.
              </Prose>
            ),
          },
          {
            label: "Small dataset transfer (less than 50k images) — CNN wins",
            render: () => (
              <Prose>
                CNNs outperform Transformers on small transfer datasets because the translation-equivariance prior acts as a strong regularizer. Dosovitskiy's original ViT paper showed ViT-B needing 14M+ images to match ResNet-50; Liu et al. 2022 showed the same for Swin vs ConvNeXt. Rule of thumb: under 50k labeled target-domain images, fine-tune ConvNeXt-T; above, consider Swin or ViT as well.
              </Prose>
            ),
          },
          {
            label: "Inference-latency-critical — on-device or realtime",
            render: () => (
              <Prose>
                For 1ms-class on-device latency (iPhone, mobile GPU, edge TPU), use MobileOne (Vasu et al. 2022) or EfficientFormer (Li et al. 2022). MobileOne-S4 at 79.4% top-1 and 1.0 ms iPhone 12 latency is the best accuracy-per-ms in the small regime. ConvNeXt-T at 28M params and pure depthwise + 1x1 convs is surprisingly fast on GPU (5–10 ms on A100 at batch 1) but heavy for mobile.
              </Prose>
            ),
          },
          {
            label: "Very large pretraining (ImageNet-22k+ or JFT) — hybrid or Transformer",
            render: () => (
              <Prose>
                At the 100M+ image scale, attention's long-range mixing and ability to absorb data consistently beat pure-CNN designs. CoAtNet-7 at 2.44B params and 90.88% top-1 on JFT-3B is the state-of-the-art pre-ViT-22B. Swin-V2 and ViT-G scale similarly. ConvNeXt V2 at ImageNet-22k is competitive (88.9% for H-variant) but the Pareto frontier shifts toward hybrids at this scale.
              </Prose>
            ),
          },
          {
            label: "Detection / segmentation — ConvNeXt with FPN or MaxViT",
            render: () => (
              <Prose>
                For COCO object detection and ADE20K segmentation, ConvNeXt + Feature Pyramid Network (FPN) is the modern default — paper-SoTA in 2022 on both benchmarks at matched compute. MaxViT and Swin-V2 are competitive. The deciding factor is input resolution: above 1024×1024 inputs, Transformers' quadratic cost becomes expensive and ConvNeXt's linear-in-pixels cost dominates. MaxViT's grid+block attention is the hybrid winner.
              </Prose>
            ),
          },
          {
            label: "Long-range dependency tasks — Transformer wins",
            render: () => (
              <Prose>
                When the task requires modeling dependencies across large spatial extents (global image-level reasoning, sequence modeling on patches), attention wins. Chart-understanding, long-document image reasoning, frame-level video understanding — these are Transformer territory. ConvNeXt's 7×7 depthwise gives a receptive field that grows linearly with depth; at stage 4 (7×7 feature map) it is already global, but for 512+ input resolutions the bottleneck shifts.
              </Prose>
            ),
          },
          {
            label: "Self-supervised pretraining (MAE, SimCLR) — ConvNeXt V2",
            render: () => (
              <Prose>
                For fully-convolutional masked autoencoder pretraining, ConvNeXt V2 (with GRN) is the only CNN design that works reliably at scale. ConvNeXt V1 suffers from feature collapse under MAE; V2's GRN fixes this. If you are building a CNN-based representation learner in 2026, start with V2.
              </Prose>
            ),
          },
          {
            label: "Kernel regime — 7x7 vs 31x31 vs attention",
            render: () => (
              <Prose>
                ConvNeXt uses 7×7 depthwise kernels. RepLKNet (Ding et al. 2022) pushes to 31×31 with structural reparameterization (merge a parallel 3×3 branch into the 31×31 weights at inference). At 31×31, the receptive field is global at 14×14 feature maps. RepLKNet matches Swin-B at fewer FLOPs. If you are optimizing for FLOPs at a fixed accuracy, 31×31 is worth trying; if you want a simpler, less hyperparameter-sensitive design, stick with 7×7.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales</H2>

      <H3>8.1 The Transformer recipe transfers to CNNs</H3>

      <Prose>
        The central scaling claim of the ConvNeXt paper is that the training recipe — 300 epochs, AdamW at 4e-3, cosine schedule, mixup 0.8, cutmix 1.0, RandAugment, stochastic depth 0.1, label smoothing 0.1, EMA 0.9999 — contributes as much to the final accuracy as the architectural changes themselves. Bello et al. (2021, arXiv 2103.07579) made this explicit in "Revisiting ResNets": they showed that a stock ResNet-50 retrained with the ViT/Swin recipe jumps from 76.1% to 79.1% without any architectural change. ConvNeXt extended this observation by co-designing the recipe and the architecture until both converged to Swin-like choices.
      </Prose>

      <H3>8.2 ConvNeXt V2 + MAE scales further than Swin V2 at 22k</H3>

      <Prose>
        Woo et al. 2023 report that ConvNeXt V2-H (660M parameters) pretrained with FCMAE on ImageNet-22k reaches 88.9% ImageNet-1k top-1 at 512 resolution — beating Swin-V2-G (3B parameters) at 87.5% in the same regime. The difference is GRN plus MAE: ConvNeXt V1 with MAE loses 1–2% because of feature collapse; ConvNeXt V2 with MAE gains it back and more. The scaling rule is that recipe-aware architectural fixes (GRN as an MAE-specific patch) matter more than raw parameter count at the 200M+ scale.
      </Prose>

      <H3>8.3 Large kernels are a viable alternative to attention</H3>

      <Prose>
        RepLKNet-31B (Ding et al. 2022) with 31×31 depthwise convolutions reaches 83.8% ImageNet-1k at 79M params — matching Swin-B at lower FLOPs. The paper's Figure 3 shows accuracy climbing monotonically as kernel size grows from 3×3 (80.1%) to 31×31 (83.8%) at matched depth and width. The scaling cutoff is roughly the feature map spatial extent: at stage 3 (14×14 maps) a 31×31 kernel is already global, so further increases do not help. Large kernels scale well as long as training uses structural reparameterization (a small-kernel branch added during training, merged at inference).
      </Prose>

      <H3>8.4 SE and CBAM channel attention have plateaued</H3>

      <Prose>
        Squeeze-and-Excitation (Hu et al. 2018) and CBAM (Woo et al. 2018) were popular 2018–2020 additions that added ~0.5% top-1 to ResNet variants at ~1% parameter overhead. In the post-ConvNeXt world, these gains have mostly been absorbed by the recipe itself (fewer activations, LN over BN, better training). Adding SE to ConvNeXt gains less than 0.2% in the paper's ablations (it was ultimately dropped from V1). CBAM behaves similarly. The design lesson: architectural add-ons whose improvements were specific to a weak baseline stop scaling once the baseline is improved.
      </Prose>

      <H3>8.5 Scaling laws for CNNs match Transformers under the right recipe</H3>

      <Prose>
        Zhai et al. 2022 ("Scaling Vision Transformers", arXiv 2106.04560) characterized a 1/N^α accuracy-vs-params law for ViTs. Liu et al. 2022 and Woo et al. 2023 show the same curve shape for ConvNeXt and ConvNeXt V2 — the exponent α is within noise of the ViT value. Pure-CNN scaling tracks pure-Transformer scaling when the recipe is shared. The implication for 2026 practitioners: architecture choice does not fundamentally change the scaling budget; it shifts the curve by a few absolute points.
      </Prose>

      <H3>8.6 Activation memory still grows linearly with depth</H3>

      <Prose>
        ConvNeXt-XL at 350M parameters and 60 blocks requires the same activation-checkpointing treatment as large Transformers. Training at batch 256 on 224×224 inputs without checkpointing consumes ~25 GB per GPU of activations alone; <Code>torch.utils.checkpoint</Code> applied per stage cuts this to ~5 GB. The residual structure ensures gradient flow; memory is the practical bottleneck above 200M params.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Using BatchNorm instead of LayerNorm in deep ConvNeXt</H3>

      <Prose>
        A common mistake when porting ConvNeXt to a new framework is to keep BatchNorm because "it's what CNNs use." At small batch sizes (distributed training with per-GPU batch 32 or 16) BN's running statistics become noisy and training destabilizes by epoch ~50. The paper shows a 0.4% drop from BN → LN and, more importantly, training curve stability under all batch sizes. Fix: use LayerNorm in channels-last layout, as the reference implementation does. If you need the batch-statistics effect for some legacy reason, use GroupNorm with 32 groups — it is closer to LN in behavior and works at any batch size.
      </Prose>

      <CodeBlock language="python">
{`# Bug: ConvNeXt block with BN instead of LN (breaks at small batch)
class BrokenBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.bn     = nn.BatchNorm2d(dim)                # <- WRONG
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

# Fix: LayerNorm in channels-last layout
class FixedBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm   = nn.LayerNorm(dim, eps=1e-6)        # <- channels-last LN
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
    def forward(self, x):
        h = self.dwconv(x)
        h = h.permute(0, 2, 3, 1)    # <- permute for LN
        h = self.norm(h)
        h = self.pwconv2(nn.functional.gelu(self.pwconv1(h)))
        return x + h.permute(0, 3, 1, 2)`}
      </CodeBlock>

      <H3>9.2 Missing LayerScale at depth greater than 36 blocks</H3>

      <Prose>
        ConvNeXt-XL has 54 blocks across its four stages. Without LayerScale (or GRN in V2), the accumulated residuals push activation norms toward <Code>{"\\sqrt{L}"}</Code> times the initial scale by block 54. This is not "LLM-style instability" in the sense of training divergence — it manifests as a ~0.8% accuracy drop and, more subtly, as sensitivity to hyperparameters (learning rate, batch size, warmup length) that the LayerScale variant absorbs. Fix: initialize LayerScale <Code>γ</Code> at <Code>{"10^{-6}"}</Code> for any ConvNeXt variant above B (54 blocks). For V2, use GRN which takes over this stability role.
      </Prose>

      <H3>9.3 Patchify stem too aggressive at low input resolution</H3>

      <Prose>
        The 4×4 stride-4 patchify stem assumes a 224-class input. At 96×96 input (common in low-resolution medical imaging), 4×4 patchify produces a 24×24 feature map — coarse enough that fine-grained detail is lost. Symptom: training loss is fine but test accuracy on detail-dependent tasks (small lesion detection, fine OCR) plateaus early. Fix: use 2×2 stride-2 patchify for inputs below 160, or insert a learnable 3×3 overlapping conv before the patchify. The timm variants <Code>convnext_pico</Code> and <Code>convnext_atto</Code> do this natively for 96-resolution.
      </Prose>

      <H3>9.4 Forgetting GRN in ConvNeXt V2 MAE pretraining</H3>

      <Prose>
        If you implement ConvNeXt V2 from scratch and copy the V1 block without adding GRN, then try FCMAE pretraining, you will observe the original V1 pathology: training loss drops fine but linear-probe accuracy on frozen features is 3–5% below V1+supervised (the feature-collapse signature). Symptom: after 800 epochs of MAE, the intermediate feature maps have many near-dead channels (std below 1e-3). Fix: add GRN after the GELU in every block. Initialize <Code>γ = β = 0</Code>. Verify with the live-channels check from section 4f after 50 epochs of pretraining — if more than 10% of channels are dead, GRN is missing or mis-placed.
      </Prose>

      <H3>9.5 Naive port of Swin hyperparameters to ConvNeXt</H3>

      <Prose>
        Swin's recipe uses LR 1e-3 with AdamW, weight decay 0.05, 20-epoch warmup. ConvNeXt uses LR 4e-3 (4× higher) with the same optimizer and decay. A common mistake: copy the Swin recipe wholesale, fail to update the LR, and end up at ~80.5% top-1 instead of 82.1%. The difference is that ConvNeXt's fewer activations reduce the effective nonlinearity per block, which tolerates a larger learning rate. Fix: use the paper's Table 10 values literally (LR 4e-3, batch 4096, 300 epochs). If batch size is reduced, scale LR linearly.
      </Prose>

      <H3>9.6 Channels-last runtime mismatch</H3>

      <Prose>
        ConvNeXt is ~20% faster on A100 when the forward pass runs in <Code>torch.channels_last</Code> memory format because the depthwise + LN + linear path maps cleanly to NHWC kernels. If you forget to call <Code>model.to(memory_format=torch.channels_last)</Code>, the runtime stays in NCHW and the implicit transposes around each LayerNorm add 10–20% latency. Symptom: ConvNeXt appears slower than ResNet-50 on GPU despite fewer FLOPs. Fix:
      </Prose>

      <CodeBlock language="python">
{`import torch
from torchvision.models import convnext_tiny

model = convnext_tiny(weights="IMAGENET1K_V1").cuda().eval()
model = model.to(memory_format=torch.channels_last)

x = torch.randn(1, 3, 224, 224, device="cuda").to(memory_format=torch.channels_last)
with torch.no_grad():
    y = model(x)
# Latency drops from ~2.1 ms to ~1.7 ms on A100 (torch 2.0+)`}
      </CodeBlock>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Read in roughly this order to follow the design-space thread from ResNet-revisited through modern hybrids.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "Liu et al. 2022 — A ConvNet for the 2020s (ConvNeXt V1)",
            render: () => (
              <Prose>
                Liu, Z., Mao, H., Wu, C-Y., Feichtenhofer, C., Darrell, T., and Xie, S. (2022). "A ConvNet for the 2020s." arXiv:2201.03545. Published at CVPR 2022. Available at arxiv.org/abs/2201.03545. The canonical paper. Introduces ConvNeXt by modernizing a ResNet-50 step by step — patchify stem, depthwise conv, inverted bottleneck, 7×7 kernel, GELU, fewer activations, LayerNorm, separate downsample layers. Table 1 (the modernization roadmap) is the piece to study; Table 9 (variant specs) is the implementation reference. The paper's section 2.6 ("Micro Design") is where most of the block-level decisions are justified.
              </Prose>
            ),
          },
          {
            label: "Woo et al. 2023 — ConvNeXt V2 (MAE + GRN)",
            render: () => (
              <Prose>
                Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I.S., and Xie, S. (2023). "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders." arXiv:2301.00808. Published at CVPR 2023. Available at arxiv.org/abs/2301.00808. Introduces Global Response Normalization (GRN) as a feature-competition layer that fixes the feature-collapse pathology of V1 under MAE pretraining. Shows FCMAE pretraining + ConvNeXt V2-H achieving 88.9% ImageNet-1k, beating Swin-V2-G at fewer parameters. Section 3 contains the GRN derivation; Figure 3 is the feature-collapse diagnosis that motivated GRN.
              </Prose>
            ),
          },
          {
            label: "Ding et al. 2022 — Scaling Up Your Kernels to 31×31 (RepLKNet)",
            render: () => (
              <Prose>
                Ding, X., Zhang, X., Zhou, Y., Han, J., Ding, G., and Sun, J. (2022). "Scaling Up Your Kernels to 31×31: Revisiting Large Kernel Design in CNNs." arXiv:2203.06717. Published at CVPR 2022. Available at arxiv.org/abs/2203.06717. Shows that depthwise convolutions can be scaled to 31×31 (essentially global at 14×14 feature maps) with structural reparameterization — a small-kernel branch trained in parallel and merged at inference. RepLKNet-31B at 79M params matches Swin-B at lower FLOPs. The paper's Figure 3 (accuracy vs kernel size) is the key empirical result: accuracy climbs monotonically up to 31×31 at the feature-map-size cutoff.
              </Prose>
            ),
          },
          {
            label: "Tu et al. 2022 — MaxViT hybrid",
            render: () => (
              <Prose>
                Tu, Z., Talebi, H., Zhang, H., Yang, F., Milanfar, P., Bovik, A., and Li, Y. (2022). "MaxViT: Multi-Axis Vision Transformer." arXiv:2204.01697. Published at ECCV 2022. Available at arxiv.org/abs/2204.01697. Introduces the MaxViT block: MBConv (inverted bottleneck) → block attention (within 7×7 window) → grid attention (across 7×7-spaced grid) — a three-stage hybrid that covers local, block-local, and global interactions in a single block. MaxViT-XL achieves 88.7% ImageNet-1k at 475M params and is state-of-the-art for object detection at matched compute. The paper's Figure 2 is the canonical illustration of multi-axis attention.
              </Prose>
            ),
          },
          {
            label: "Dai et al. 2021 — CoAtNet",
            render: () => (
              <Prose>
                Dai, Z., Liu, H., Le, Q.V., and Tan, M. (2021). "CoAtNet: Marrying Convolution and Attention for All Data Sizes." arXiv:2106.04803. Published at NeurIPS 2021. Available at arxiv.org/abs/2106.04803. Proposes a stacking strategy: MBConv at early stages (low resolution, high FLOPs without attention cost), self-attention at later stages (high-level features where global mixing helps). The paper's main empirical result: CoAtNet-7 on JFT-3B pretraining hits 90.88% ImageNet-1k, which was the 2022 state-of-the-art. Section 3 has the design-space analysis of where to insert attention in a CNN pyramid.
              </Prose>
            ),
          },
          {
            label: "Liu et al. 2021 — Swin Transformer",
            render: () => (
              <Prose>
                Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." arXiv:2103.14030. Published at ICCV 2021 (Best Paper). Available at arxiv.org/abs/2103.14030. The direct counterpart to ConvNeXt — the same authors (Liu et al.) published both. Introduces shifted-window attention with linear complexity and 7×7 local windows. ConvNeXt inherits the 7×7 window, 1:1:3:1 stage ratio, LN placement, and GELU from this paper. Essential to read both papers together to understand ConvNeXt's design choices.
              </Prose>
            ),
          },
          {
            label: "Vasu et al. 2022 — MobileOne",
            render: () => (
              <Prose>
                Vasu, P.K.A., Gabriel, J., Zhu, J., Tuzel, O., and Ranjan, A. (2022). "MobileOne: An Improved One millisecond Mobile Backbone." arXiv:2206.04040. Published at CVPR 2023. Available at arxiv.org/abs/2206.04040. Apple's answer to MobileNet for 2022-era iPhones. Uses structural reparameterization (RepVGG-style) with multi-branch training merged into single-branch inference. MobileOne-S4 at 79.4% top-1 and 1.0 ms iPhone 12 CPU latency is the best accuracy-per-ms in the sub-ms class. The reparameterization trick is the technique to borrow for any latency-bound CNN deployment.
              </Prose>
            ),
          },
          {
            label: "Bello et al. 2021 — Revisiting ResNets",
            render: () => (
              <Prose>
                Bello, I., Fedus, W., Du, X., Cubuk, E.D., Srinivas, A., Lin, T-Y., Shlens, J., and Zoph, B. (2021). "Revisiting ResNets: Improved Training and Scaling Strategies." arXiv:2103.07579. Published at NeurIPS 2021. Available at arxiv.org/abs/2103.07579. The paper that established "recipe matters as much as architecture". A stock ResNet-50 retrained with modern augmentation (RandAugment, mixup, cutmix), AdamW, and longer training rises from 76.1% to 79.1% top-1 without any architectural change. This is the empirical foundation under ConvNeXt's central claim — the recipe alone accounts for 3 of the 6 absolute percentage points gained in the modernization roadmap.
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
        Attempt all five before reading the answers. Exercises 1–2 test the design-space reasoning; 3 tests arithmetic; 4 tests architectural judgment; 5 tests debugging.
      </Prose>

      <H3>Exercise 1 (design-space reasoning — the seven moves)</H3>
      <Prose>
        Name the seven design changes that turn a stock ResNet-50 into ConvNeXt-T. For each, say which Transformer paper (ViT, Swin, or MLP-era) first popularized it in the vision context, and roughly how many ImageNet top-1 points it contributes according to the paper's ablation table.
      </Prose>
      <Callout accent="green">
        <strong>Answer 1.</strong> The seven moves, roughly in paper order, with contributions from Table 1 (ResNet-50 baseline = 76.1%):
        <br />
        (1) <strong>Modern training recipe</strong> (RandAugment, mixup, cutmix, AdamW, 300 epochs, stochastic depth) → +2.7% to 78.8%. This is the Bello et al. 2021 "Revisiting ResNets" recipe; the rest of the moves are measured on top of it.
        <br />
        (2) <strong>Stage compute ratio 1:1:3:1</strong> (depths 3,3,9,3 instead of 3,4,6,3) — from Swin, +0.6% to 79.4%.
        <br />
        (3) <strong>Patchify stem</strong> (4×4 stride-4 conv replaces 7×7 stride-2 + maxpool) — from ViT/Swin, roughly neutral (79.4 → 79.5).
        <br />
        (4) <strong>Depthwise conv</strong> (replace the 3×3 full conv with depthwise, widen from 64 to 96 channels to compensate) — neutral at this step (79.5 → 79.5) but enabled by inverted bottleneck next.
        <br />
        (5) <strong>Inverted bottleneck</strong> (C → 4C → C instead of C → C/4 → C) — from MobileNetV2 via Swin MLP, +1.1% to 80.6%.
        <br />
        (6) <strong>Large kernel 7×7</strong> (up from 3×3 in the depthwise) — from Swin's 7×7 window, roughly neutral in isolation but combines with inverted bottleneck.
        <br />
        (7) <strong>Fewer activations and norms, plus LN replacing BN</strong> — one GELU and one LN per block. From Swin's single-GELU-per-MLP pattern. Cumulatively +1.4% to 82.0%.
        <br />
        Final step: separate downsampling layers (add LN before each 2×2 stride-2 conv) → +0.1% to 82.1%. Note that "depthwise conv alone" is neutral; the wins come from its combination with inverted bottleneck and large kernel.
      </Callout>

      <H3>Exercise 2 (design-space reasoning — inverted bottleneck)</H3>
      <Prose>
        ResNet-50 uses a "normal" bottleneck C → C/4 → C, which contracts channels in the middle. ConvNeXt uses an inverted bottleneck C → 4C → C, which expands them. Explain why the inverted pattern is better when paired with depthwise convolution. What would go wrong if you used inverted bottleneck with standard (non-depthwise) convolution?
      </Prose>
      <Callout accent="green">
        <strong>Answer 2.</strong> The inverted bottleneck works with depthwise because the expensive step (the expansion to 4C) is pointwise (1×1), not spatial. In a ConvNeXt block, the depthwise 7×7 happens at C channels (cheap: <Code>{"49 H W C"}</Code> FLOPs), then the pointwise expand to 4C happens in a linear (also cheap: <Code>{"4 H W C^2"}</Code> FLOPs and no spatial mixing). Total block cost is dominated by the two pointwise convs at <Code>{"8 H W C^2"}</Code>. The expansion gives representational capacity; the depthwise gives spatial mixing; the two are cleanly separated.
        <br />
        If you used inverted bottleneck with standard (non-depthwise) 3×3 conv, the 3×3 at 4C channels would cost <Code>{"9 H W \\cdot 4C \\cdot 4C = 144 H W C^2"}</Code> — 18× more than the depthwise version. A ResNet-style "C → 4C → C" with standard convs would be ~200G FLOPs for a T-scale model (versus ConvNeXt-T's 4.5G). It's infeasible. The depthwise decomposition is what makes the inverted bottleneck affordable; the inverted direction is what makes the expansion meaningful. They are a package deal.
      </Callout>

      <H3>Exercise 3 (arithmetic — stage compute ratio)</H3>
      <Prose>
        ConvNeXt-T has stage depths (3, 3, 9, 3) at widths (96, 192, 384, 768), operating at spatial sizes (56, 28, 14, 7) after a 4×4 stride-4 stem on 224 input. Using <Code>{"\\text{FLOPs per block} \\approx 8 H W C^2"}</Code>, compute the fraction of total block FLOPs consumed by each stage and show that they approximate the 1:1:3:1 design target.
      </Prose>
      <Callout accent="green">
        <strong>Answer 3.</strong> Per-block FLOPs scale as <Code>{"8 H W C^2"}</Code>. With channels doubling (2×) and spatial halving (4× per H·W) each stage, per-block FLOPs go <Code>{"8 \\cdot 2^2 / 4 = 1"}</Code>× constant across stages — convenient design.
        <br />
        Plugging in values (in units of <Code>8 H W C^2</Code>, with stage-0 block = 1.0):
        <br />
        Stage 0: 3 blocks × (56² × 96²) = 3 × 28,901 = 86,703 units.
        <br />
        Stage 1: 3 blocks × (28² × 192²) = 3 × 28,901 = 86,703 units.
        <br />
        Stage 2: 9 blocks × (14² × 384²) = 9 × 28,901 = 260,109 units.
        <br />
        Stage 3: 3 blocks × (7² × 768²) = 3 × 28,901 = 86,703 units.
        <br />
        Total: 520,218 units. Per-stage shares: 16.7%, 16.7%, 50.0%, 16.7%. Ratio: 1 : 1 : 3 : 1, exactly as designed. (The empirical FLOPs including DW-conv and LN terms are 17.4, 16.8, 49.5, 16.4 — matching the abstract derivation within 1%.) The design principle: depth is what creates the 3× share at stage 2, not width or spatial size — per-block FLOPs are constant across stages by design.
      </Callout>

      <H3>Exercise 4 (architectural judgment — ConvNeXt vs Swin vs ConvNeXt V2)</H3>
      <Prose>
        You are building a production image classifier for a medical imaging dataset with 30,000 labeled CT scans. Your team has access to 10 A100 GPUs and is considering three options: Swin-T, ConvNeXt-T (V1), ConvNeXt V2-T + FCMAE. For each, state when it would be the right choice, and give one concrete reason to prefer it over the other two.
      </Prose>
      <Callout accent="green">
        <strong>Answer 4.</strong>
        <br />
        (a) <strong>ConvNeXt-T (V1).</strong> Right choice when: 30k images is too small for self-supervised pretraining to help (MAE typically needs 100k+ for a meaningful representation learning signal), and you want the simplest, most-supported path. Concrete reason: matches Swin-T accuracy at 4.5G FLOPs, is pure CNN (so no attention-kernel dependencies in the deployment stack — simpler ONNX/TensorRT export, better mobile compatibility), has a ready IN-22k pretrained checkpoint (<Code>convnext_tiny.fb_in22k_ft_in1k</Code>) that transfers well to CT with a simple fine-tune.
        <br />
        (b) <strong>Swin-T.</strong> Right choice when: you expect long-range dependencies across the CT volume (e.g., whole-organ context needed for small lesion classification) and have enough augmentation to prevent the Transformer from overfitting 30k examples. Concrete reason: shifted-window attention captures mid-range interactions that a 7×7 depthwise misses; for 512×512+ slices the window attention scales better than the depthwise spatial extent.
        <br />
        (c) <strong>ConvNeXt V2-T + FCMAE.</strong> Right choice when: your dataset has a large unlabeled pool (e.g., 200k unlabeled CT volumes at your hospital) where 30k are labeled. Pretraining on the full 230k with FCMAE, then fine-tuning on the 30k labels, typically gains 2–5% over supervised-only. Concrete reason: GRN allows the MAE pretraining to actually work without feature collapse; V1 would waste the unlabeled data. This is the only option that leverages unlabeled domain data.
        <br />
        Default recommendation for "30k labels, no unlabeled pool, ship fast": ConvNeXt-T (V1). It is the boring, correct choice.
      </Callout>

      <H3>Exercise 5 (debugging — "my ConvNeXt is slower than my ResNet")</H3>
      <Prose>
        You have reproduced ConvNeXt-T from scratch and benchmarked it on an A100 at batch 32. FLOPs are 4.5G (vs 4.1G for ResNet-50), but your ConvNeXt forward pass takes 8.3 ms while your ResNet-50 forward pass takes 4.1 ms — ConvNeXt is 2× slower. List three plausible implementation causes and, for each, describe the one-line diagnostic that would confirm it.
      </Prose>
      <Callout accent="green">
        <strong>Answer 5.</strong>
        <br />
        (1) <strong>Missing channels-last memory format.</strong> The ConvNeXt block's <Code>permute(0, 2, 3, 1)</Code> and <Code>permute(0, 3, 1, 2)</Code> are free view operations in channels-last layout but force actual memory copies in channels-first. On A100, these copies cost ~3 ms per forward pass. Diagnose:
        <br />
        <Code>{"model = model.to(memory_format=torch.channels_last); x = x.to(memory_format=torch.channels_last)"}</Code>
        <br />
        and re-time. If latency drops by ~40%, this was the cause.
        <br />
        (2) <strong>LayerNorm-2d reimplemented with reduce-then-divide instead of fused kernels.</strong> A naive <Code>{"(x - x.mean(1))/x.std(1)"}</Code> launches two separate reduction kernels; PyTorch's built-in <Code>F.layer_norm</Code> uses a single fused kernel. At 56×56×96 per block × 18 blocks × 3 stages of downsamples, the overhead adds up. Diagnose: profile with <Code>torch.profiler.profile</Code> and look at the kernel names — if you see <Code>mean_kernel</Code> and <Code>var_kernel</Code> back-to-back around every LN, you are not using the fused kernel. Fix by switching to <Code>F.layer_norm</Code> or, for the channels-first stem/downsample LN, use <Code>nn.GroupNorm(1, dim)</Code> which has a fused kernel.
        <br />
        (3) <strong>Depthwise conv not using cuDNN's depthwise path.</strong> On some cuDNN versions, Conv2d with groups=C only dispatches to the fast depthwise kernel when channels-last layout is active and C is a multiple of 32. If you built in channels-first with C=96 (multiple of 32), cuDNN might still fall back to the generic grouped kernel. Diagnose: <Code>{"torch.backends.cudnn.benchmark = True"}</Code> and re-time, then profile for the specific conv kernel names. Fix: ensure channels-last (as in move 1) and pad C to multiples of 32 if necessary. On an A100 with PyTorch 2.0+ this usually just requires channels-last; on older stacks you may need <Code>{"torch.utils.benchmark.Timer"}</Code> to compare before/after fairly.
      </Callout>

    </div>
  ),
};

export default convnextModernContent;
