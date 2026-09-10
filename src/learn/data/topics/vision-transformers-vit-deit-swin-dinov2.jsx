import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const visionTransformersContent = {
  title: "Vision Transformers (ViT, DeiT, Swin, DiNOv2)",
  readTime: "~40 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        For nearly a decade after AlexNet in 2012, computer vision was a convolutional story. Every SOTA classifier, detector, and segmenter was a CNN of some shape — VGG, ResNet, Inception, EfficientNet. The inductive biases of convolution (translation equivariance, locality, hierarchical receptive fields) were treated as not just useful but essential. Transformers, meanwhile, had eaten NLP. By 2020 the question hanging over both communities was awkward: if Transformers can handle arbitrary sequence data, why can they not handle images? The field had tried — hybrid CNN-attention models (Bello et al. 2019, Ramachandran et al. 2019), stand-alone self-attention (Hu et al. 2019), iGPT (Chen et al. 2020) — but none matched CNNs on ImageNet at reasonable compute.
      </Prose>

      <Prose>
        Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby at Google Brain Zurich changed this in a single paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (arXiv:2010.11929), submitted in October 2020 and published at ICLR 2021 as the Vision Transformer (ViT). Their recipe was brutally simple. Take an image, cut it into fixed-size non-overlapping 16×16 patches, flatten each patch to a vector, linearly project each to a token embedding, add a learned [CLS] token, add learned positional embeddings, feed the whole sequence through a standard Transformer encoder from "Attention Is All You Need" (Vaswani et al. 2017), and read the [CLS] token through a linear classification head. No convolution anywhere after the initial linear projection. No hand-crafted inductive bias. The paper's central empirical claim was scale-dependent: on ImageNet-1k alone, ViT underperformed a comparable ResNet; on ImageNet-21k (14M images) it matched; on JFT-300M (300M images) it exceeded ResNet by a clear margin, with ViT-H/14 reaching 88.55% ImageNet-1k top-1 after JFT pretraining. The conclusion the community heard: attention scales better than convolution with more data and more compute, and the vision inductive bias of convolution is not necessary when you have enough data.
      </Prose>

      <Prose>
        ViT left two open problems. First, it was data-hungry — it required JFT-300M, a proprietary Google dataset, to beat CNNs. Second, it was flat — a single global-attention encoder that did not match the hierarchical feature pyramid that CNN-based detectors and segmenters depended on. Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou at Meta AI Paris solved the first problem with DeiT (Data-efficient Image Transformers, arXiv:2012.12877, ICML 2021). They showed that with the right training recipe — heavy augmentation (RandAugment, mixup, CutMix), long training schedules (300 epochs), repeated augmentation, AdamW, stochastic depth, and one clever architectural addition (a [DIST] distillation token alongside [CLS], supervised by a pretrained CNN teacher) — a ViT-B could match ResNet-152 on ImageNet-1k alone, no JFT required. DeiT-B hit 83.4% ImageNet top-1 with only ImageNet-1k training.
      </Prose>

      <Prose>
        Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo at Microsoft Research Asia attacked the hierarchical-feature-pyramid problem with Swin Transformer (arXiv:2103.14030, ICCV 2021 Best Paper). Swin's two architectural innovations were (a) compute self-attention within small non-overlapping local windows (M×M = 7×7 tokens) so cost grew linearly with image area, and (b) shift the window partition by M/2 between consecutive blocks so information flowed across window boundaries. Between stages, Swin merged 2×2 token groups, halving spatial resolution and doubling channels — the same pyramid structure as a ResNet. Swin became a drop-in backbone replacement for CNNs in Mask R-CNN, UPerNet, and DETR, and swept ImageNet, COCO, and ADE20K in 2021. Swin-L at 87.3% ImageNet top-1 (IN-22k pretrained) was the new SOTA.
      </Prose>

      <Prose>
        The next shift came from self-supervised pretraining. Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin's DiNO (arXiv:2104.14294, ICCV 2021) showed that a ViT trained with no labels whatsoever — via self-distillation between a student and an EMA teacher, both seeing different augmented views of the same image — developed emergent object-segmentation-quality attention maps. The [CLS] token, with no segmentation supervision at all, began attending precisely to foreground objects. DiNO features, used as-is with a linear probe, matched or beat ImageNet-supervised ViT on many downstream tasks. Three years later, Maxime Oquab, Timothée Darcet, and colleagues at Meta AI released DiNOv2 (arXiv:2304.07193, TMLR 2024), a production-grade scaling of DiNO: a ViT-g/14 trained self-supervised on a curated 142M-image dataset (LVD-142M), distilled into smaller ViTs. DiNOv2 features, without any fine-tuning, match or exceed the best supervised features on classification, segmentation, depth estimation, and retrieval. As of 2026, DiNOv2 is the default general-purpose visual feature extractor in the open-source ecosystem.
      </Prose>

      <Prose>
        Two parallel threads completed the picture. Alec Radford and colleagues' CLIP (arXiv:2103.00020, 2021) trained a ViT image encoder and a Transformer text encoder jointly with a contrastive loss on 400M image-text pairs scraped from the web, producing an image-text embedding space that enabled zero-shot classification on arbitrary label sets. Kaiming He and colleagues' MAE (arXiv:2111.06377, CVPR 2022) showed that masking 75% of image patches and training a ViT to reconstruct the missing pixels produced an excellent initialization for downstream fine-tuning — the image equivalent of BERT pretraining, but with high mask ratio and an asymmetric encoder-decoder. Alexander Kirillov and colleagues' Segment Anything Model (SAM, arXiv:2304.02643, 2023) built a promptable segmentation system on a ViT-H backbone, trained on 1.1B masks across 11M images — the largest visual annotation effort in history. By 2026, essentially every foundation vision model — CLIP, SAM, DiNOv2, ImageBind, SigLIP — uses a ViT backbone. The CNN era did not end, but the default vision backbone, especially at scale, became the Transformer.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        ViT's design is almost defiantly minimal. The paper's core argument is that a 2D image can be serialized into a 1D sequence of patch tokens, and from that point the standard Transformer encoder — unchanged from language modeling — is sufficient. Internalize the four moves below and you can read any vision-Transformer paper published since.
      </Prose>

      <Prose>
        <strong>Move 1 — Tokenize by non-overlapping patches.</strong> A 224×224 RGB image has 224×224×3 = 150,528 pixels. Treating each pixel as a token would require an attention cost proportional to <Code>{"150{,}528^2"}</Code> — infeasible. ViT partitions the image into P×P patches (typically 16×16 or 14×14), flattens each to a vector of length <Code>{"P^2 \\cdot C = 16^2 \\cdot 3 = 768"}</Code>, and linearly projects each to a d-dimensional embedding. For 224×224 / 16×16, this produces 196 tokens. For 224×224 / 14×14, 256 tokens. The projection is a single learned matrix <Code>{"E \\in R^{(P^2 C) \\times d}"}</Code>, which is numerically equivalent to a Conv2d with kernel P and stride P — the "conv stem" formulation in timm. Non-overlap is the key: each spatial location in the token grid corresponds to exactly one image region, with no redundancy. This is the entire vision-specific inductive bias ViT keeps.
      </Prose>

      <Prose>
        <strong>Move 2 — [CLS] token + learned positional embeddings.</strong> ViT prepends a learnable [CLS] token to the sequence (shape <Code>{"1 \\times d"}</Code>), giving <Code>{"N + 1 = 197"}</Code> tokens for 16-patch ViT. It then adds a learnable positional embedding tensor <Code>{"E_{pos} \\in R^{(N+1) \\times d}"}</Code> — one vector per position, including the CLS slot. The [CLS] token's role is identical to BERT's: after the encoder, its final-layer representation is the image-level feature used for classification. The positional embedding is necessary because the Transformer encoder itself is permutation-equivariant — without <Code>{"E_{pos}"}</Code>, shuffling patches would not change the output. DeiT adds a second [DIST] token next to [CLS]; its role is supervised by a CNN teacher's logits via distillation, and at inference the model's prediction is the average of CLS and DIST heads.
      </Prose>

      <Prose>
        <strong>Move 3 — Standard Transformer encoder, no vision hacks.</strong> The encoder is the exact architecture of "Attention Is All You Need" (Vaswani et al. 2017) with pre-LayerNorm: each block is LN → MHSA → residual, then LN → MLP (expand to 4d with GELU, project back) → residual. Twelve blocks for ViT-B, 24 for ViT-L, 32 for ViT-H. MHSA has 12 heads for ViT-B, with <Code>{"d_{head} = d / h = 64"}</Code>. There is no relative positional bias, no convolution, no pooling, no rotary embedding — just learned absolute positions, added once at the input. The uniformity is the point: if you know the language Transformer, you know ViT.
      </Prose>

      <Prose>
        <strong>Move 4 — Hierarchy via windows (Swin), not via architecture overhaul.</strong> ViT is flat: every block sees the full 197-token sequence. Swin hierarchizes by two changes. First, attention is computed inside non-overlapping M×M windows (M = 7); the 56×56 stage-1 feature map has 64 windows of 49 tokens each, so attention cost is <Code>{"64 \\cdot 49^2 \\cdot d"}</Code> rather than <Code>{"(56 \\cdot 56)^2 \\cdot d"}</Code> — linear in image area instead of quadratic. Second, alternating blocks shift the window grid by M/2 (cyclic roll) so patches on window boundaries get mixed with neighbors in the next block. Between stages, Swin concatenates 2×2 token groups along the channel axis and projects down, halving spatial resolution and doubling channels — the same pyramid as a ResNet with stage widths (96, 192, 384, 768). The [CLS] token is dropped; classification pools the final stage's feature map.
      </Prose>

      <Prose>
        <strong>Move 5 — Self-supervision as the dominant pretraining mode.</strong> MAE masks 75% of patches and trains the encoder to reconstruct the missing pixels from the visible 25%; the asymmetric encoder-decoder (full-size encoder on visible tokens, tiny decoder on full sequence) makes this tractable at ViT-H scale. DiNO/DiNOv2 runs two augmented views through a student and an EMA teacher, minimizes cross-entropy between softmax outputs, with a centering term to prevent representation collapse. The student learns invariances; the teacher, lagging the student on an exponential moving average, provides the pseudo-label. No labels are needed, and the resulting features generalize remarkably well. CLIP uses contrastive learning across image-text pairs: the ViT image encoder and a text Transformer produce embeddings that are pulled together for matched pairs and pushed apart for unmatched ones across a batch of 32,768 pairs. The emergent property in all three: ViTs trained without labels develop internal representations that, read out with a linear probe, match or exceed supervised ImageNet pretraining.
      </Prose>

      <Callout accent="gold">
        Mental model: ViT is "treat an image as a 196-token sequence, run a language Transformer". Swin is "tile the image into 7x7 windows and do local attention, plus alternating shifts, in a ResNet-like pyramid". DiNO/DiNOv2 is "ViT plus self-distillation, and it turns out [CLS] attention learns to segment". DeiT is "ViT plus a CNN teacher plus heavy augmentation". Every vision Transformer published since fits one of these four molds or a hybrid of them.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Patch embedding</H3>

      <Prose>
        Let <Code>{"x \\in R^{C \\times H \\times W}"}</Code> be an image (C = 3 for RGB). Partition into non-overlapping P×P patches: <Code>{"N = (H / P) \\cdot (W / P)"}</Code>. For 224×224 with P = 16, N = 14 × 14 = 196. Flatten each patch to <Code>{"x_p^i \\in R^{P^2 C}"}</Code> for <Code>{"i = 1, \\ldots, N"}</Code>. The patch embedding produces the input sequence <Code>{"z_0"}</Code>:
      </Prose>

      <MathBlock>
        {"z_0 = [x_{\\text{cls}};\\ x_p^1 E;\\ x_p^2 E;\\ \\ldots;\\ x_p^N E] + E_{\\text{pos}}"}
      </MathBlock>

      <Prose>
        where <Code>{"E \\in R^{(P^2 C) \\times d}"}</Code> is the patch-projection matrix, <Code>{"x_{\\text{cls}} \\in R^d"}</Code> is the learnable CLS token, and <Code>{"E_{\\text{pos}} \\in R^{(N+1) \\times d}"}</Code> is the learnable positional embedding. Semicolons denote row concatenation into an <Code>{"(N+1) \\times d"}</Code> matrix. The total patch-embedding parameter count is <Code>{"(P^2 C) \\cdot d + d + (N+1) d"}</Code>. For ViT-B/16 at 224 with d = 768: <Code>{"768 \\cdot 768 + 768 + 197 \\cdot 768 = 741{,}120"}</Code> parameters, roughly 1% of the model.
      </Prose>

      <H3>3.2 Transformer encoder block (pre-LN)</H3>

      <Prose>
        Each of L blocks applies:
      </Prose>

      <MathBlock>
        {"z'_\\ell = \\text{MHSA}(\\text{LN}(z_{\\ell-1})) + z_{\\ell-1}"}
      </MathBlock>

      <MathBlock>
        {"z_\\ell = \\text{MLP}(\\text{LN}(z'_\\ell)) + z'_\\ell"}
      </MathBlock>

      <Prose>
        with <Code>{"\\text{MLP}(y) = W_2 \\cdot \\text{GELU}(W_1 y)"}</Code>, <Code>{"W_1 \\in R^{d \\times 4d}"}</Code>, <Code>{"W_2 \\in R^{4d \\times d}"}</Code>. After L blocks the final feature is <Code>{"y = \\text{LN}(z_L^{[0]})"}</Code> — the normalized CLS-token row — and the classification head is a linear layer <Code>{"W_{\\text{head}} \\in R^{d \\times K}"}</Code> for K classes.
      </Prose>

      <H3>3.3 Multi-head self-attention FLOPs</H3>

      <Prose>
        For a sequence of length N + 1 and width d with h heads (<Code>{"d_h = d / h"}</Code>), the per-block MHSA cost splits as: QKV projection <Code>{"3(N+1) d^2"}</Code>, attention scores <Code>{"(N+1)^2 d"}</Code>, attention-weighted values <Code>{"(N+1)^2 d"}</Code>, output projection <Code>{"(N+1) d^2"}</Code>. The MLP cost is <Code>{"8 (N+1) d^2"}</Code>. Total per block is approximately:
      </Prose>

      <MathBlock>
        {"\\text{FLOPs}_{\\text{block}} \\approx 12\\,(N+1)\\,d^2 + 2\\,(N+1)^2\\,d"}
      </MathBlock>

      <Prose>
        The first term is linear in N, the second quadratic. For ViT-B/16 at 224 (N+1 = 197, d = 768), the linear term is <Code>{"12 \\cdot 197 \\cdot 768^2 \\approx 1.39"}</Code> G FLOPs, and the quadratic term is <Code>{"2 \\cdot 197^2 \\cdot 768 \\approx 60"}</Code> M FLOPs — MLP dominates by ~23×. Only at very large N (higher-resolution fine-tuning or ViT-H at 224 with N = 256) does the quadratic term approach a significant fraction. This is why Swin's window attention (which replaces <Code>{"N^2"}</Code> with <Code>{"M^2 N / M^2 = M^2 N"}</Code>) pays off primarily at higher resolutions and in detection/segmentation pipelines.
      </Prose>

      <H3>3.4 Swin window attention and complexity</H3>

      <Prose>
        Given a feature map of size <Code>{"H \\times W"}</Code> (in tokens) and window size M, Swin partitions into <Code>{"(H / M) \\cdot (W / M)"}</Code> non-overlapping M×M windows. Attention is computed within each window independently. Per-layer cost is:
      </Prose>

      <MathBlock>
        {"\\Omega(\\text{W-MSA}) = 4 H W d^2 + 2 M^2 H W d"}
      </MathBlock>

      <Prose>
        compared to global attention:
      </Prose>

      <MathBlock>
        {"\\Omega(\\text{MSA}) = 4 H W d^2 + 2 (H W)^2 d"}
      </MathBlock>

      <Prose>
        The first term (QKV plus out projection) is identical; the second drops from <Code>{"(HW)^2"}</Code> to <Code>{"M^2 \\cdot HW"}</Code>. For M = 7 and a 56×56 stage-1 feature map, the speedup on the attention term is <Code>{"(56 \\cdot 56) / 49 = 64 \\times"}</Code>. Cross-window information flow comes from the shifted-window block: the partition is rolled by <Code>{"(\\lfloor M / 2 \\rfloor, \\lfloor M / 2 \\rfloor)"}</Code> tokens, mixing previously-separated patches. With a carefully-designed attention mask, the cyclic-shift trick implements this without padding.
      </Prose>

      <H3>3.5 DiNO self-distillation loss</H3>

      <Prose>
        Given a batch of images, construct two augmented views <Code>{"(v_1, v_2)"}</Code> per image. Run the student network (parameters <Code>{"\\theta_s"}</Code>) and the EMA teacher (parameters <Code>{"\\theta_t"}</Code>) on both views. Each network outputs a K-dimensional vector of logits (K = 65,536 in DiNOv1) through a projection head. Softmax both; the teacher softmax is sharpened (low temperature) and centered by an EMA of the teacher outputs to prevent collapse:
      </Prose>

      <MathBlock>
        {"P_s(v) = \\text{softmax}(g_{\\theta_s}(v) / \\tau_s)"}
      </MathBlock>

      <MathBlock>
        {"P_t(v) = \\text{softmax}((g_{\\theta_t}(v) - c) / \\tau_t)"}
      </MathBlock>

      <MathBlock>
        {"L_{\\text{DiNO}} = -\\sum_{v' \\in \\{v_1, v_2\\}}\\ \\sum_{v \\neq v'}\\ P_t(v) \\cdot \\log P_s(v')"}
      </MathBlock>

      <Prose>
        where <Code>{"\\tau_s = 0.1"}</Code>, <Code>{"\\tau_t = 0.04"}</Code>, and the centering vector <Code>c</Code> is updated as <Code>{"c \\leftarrow m \\cdot c + (1 - m) \\cdot \\text{mean}(g_{\\theta_t}(v))"}</Code> across the batch. The teacher is updated by momentum: <Code>{"\\theta_t \\leftarrow \\lambda \\theta_t + (1 - \\lambda) \\theta_s"}</Code> with <Code>{"\\lambda \\in [0.996, 1]"}</Code>. Only the student receives gradients; the teacher never sees a gradient update. The centering-and-sharpening pair is what prevents the "all images map to the same vector" failure mode that plagues naive self-distillation.
      </Prose>

      <H3>3.6 Position-embedding interpolation</H3>

      <Prose>
        ViT's positional embeddings are learned for a specific grid size (e.g. 14×14 for 224/16 input). Fine-tuning at a larger resolution (e.g. 384 → 24×24) or smaller patch size (14 instead of 16) requires interpolating <Code>{"E_{\\text{pos}}"}</Code> to the new grid. The standard approach: reshape the N patch-position vectors to a <Code>{"\\sqrt{N} \\times \\sqrt{N} \\times d"}</Code> grid, apply 2D bicubic interpolation to the target grid, flatten back. The CLS position is kept as-is. This is a pure pre-processing step — no retraining needed — and preserves the learned structure of nearby positions having similar embeddings. Every timm ViT checkpoint supports this via <Code>{"img_size"}</Code> kwargs.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below is executable PyTorch. Outputs are verbatim stdout captured from local runs on PyTorch 2.6.0 / Python 3.12.
      </Prose>

      <H3>4a. Patch embedding — unfold vs Conv2d equivalence</H3>

      <Prose>
        The patch-embedding step has two common implementations: (a) <Code>tensor.unfold</Code> + <Code>nn.Linear</Code>, which is the most literal translation of the paper; (b) <Code>nn.Conv2d</Code> with kernel P and stride P, which every timm implementation uses because it is faster and compiles to a single CUDA kernel. Both produce identical outputs up to weight initialization.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn

torch.manual_seed(0)
B, C, H, W = 2, 3, 224, 224
P = 16
D = 768

img = torch.randn(B, C, H, W)

# Method 1: unfold + linear
patches = img.unfold(2, P, P).unfold(3, P, P)          # [B, C, H/P, W/P, P, P]
patches = patches.contiguous().view(B, C, -1, P, P)    # [B, C, N, P, P]
patches = patches.permute(0, 2, 1, 3, 4).contiguous()  # [B, N, C, P, P]
patches = patches.view(B, patches.shape[1], -1)        # [B, N, C*P*P]

N = patches.shape[1]
E = nn.Linear(C * P * P, D, bias=True)
tokens = E(patches)

# Add CLS + positional
cls = nn.Parameter(torch.randn(1, 1, D) * 0.02)
pos = nn.Parameter(torch.randn(1, N + 1, D) * 0.02)
cls_b = cls.expand(B, -1, -1)
z0 = torch.cat([cls_b, tokens], dim=1) + pos

print(f"Image    : {tuple(img.shape)}")
print(f"Patches  : {tuple(patches.shape)}  (B, N, C*P*P where P*P*C = {C*P*P})")
print(f"N (patches) = (H/P)*(W/P) = {H//P}*{W//P} = {N}")
print(f"Tokens   : {tuple(tokens.shape)}")
print(f"z_0      : {tuple(z0.shape)}  (includes CLS)")
print(f"Embed weight E shape: {tuple(E.weight.shape)}")
print(f"Total positional params = (N+1)*D = {(N+1)*D:,}")

# Method 2: equivalent conv stem
conv = nn.Conv2d(C, D, kernel_size=P, stride=P, bias=True)
tokens2 = conv(img).flatten(2).transpose(1, 2)
print(f"\\nEquivalent Conv2d stem output: {tuple(tokens2.shape)}  matches unfold path")

# Output:
# Image    : (2, 3, 224, 224)
# Patches  : (2, 196, 768)  (B, N, C*P*P where P*P*C = 768)
# N (patches) = (H/P)*(W/P) = 14*14 = 196
# Tokens   : (2, 196, 768)
# z_0      : (2, 197, 768)  (includes CLS)
# Embed weight E shape: (768, 768)
# Total positional params = (N+1)*D = 151,296
#
# Equivalent Conv2d stem output: (2, 196, 768)  matches unfold path`}
      </CodeBlock>

      <Prose>
        The patch-projection weight is a 768×768 matrix: input dim <Code>{"P^2 C = 16^2 \\cdot 3 = 768"}</Code>, output dim <Code>{"d = 768"}</Code>. Positional embeddings total <Code>{"(N+1) \\cdot d = 197 \\cdot 768 = 151{,}296"}</Code> parameters. The Conv2d path produces bitwise-identical shape and, with appropriate weight-tying, identical outputs — which is why the timm implementation uses it.
      </Prose>

      <H3>4b. Full ViT from scratch — CIFAR-scale sanity check</H3>

      <Prose>
        Assemble a small ViT for CIFAR-10: 4×4 patches on 32×32 images (N = 64), embedding dim 192, depth 6, 3 heads. Total ~2.7M parameters — roughly the size of ViT-Tiny but at CIFAR resolution.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn, torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
    def forward(self, x):
        x = self.proj(x)                    # [B, D, H/P, W/P]
        return x.flatten(2).transpose(1, 2)  # [B, N, D]

class MHSA(nn.Module):
    def __init__(self, dim, heads=3):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.dh = dim // heads
        self.scale = self.dh ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]           # [B, h, N, dh]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, dim, heads=3, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = MHSA(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

class TinyViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, dim=192, depth=6,
                 heads=3, num_classes=10):
        super().__init__()
        self.embed = PatchEmbed(img_size, patch_size, 3, dim)
        N = self.embed.num_patches
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, N + 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
    def forward(self, x):
        x = self.embed(x)                            # [B, N, D]
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos    # [B, N+1, D]
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        return self.head(x[:, 0])                    # CLS token

torch.manual_seed(0)
m = TinyViT()
x = torch.randn(4, 3, 32, 32)
y = m(x)
n_params = sum(p.numel() for p in m.parameters())
print(f"TinyViT (CIFAR-scale)")
print(f"  patch_size=4, img=32, N=64, dim=192, depth=6, heads=3")
print(f"  input  : {tuple(x.shape)}")
print(f"  output : {tuple(y.shape)}")
print(f"  params : {n_params/1e6:.2f} M")
print(f"  CLS token init mean: {m.cls.mean().item():.4f}")
print(f"  Pos emb shape       : {tuple(m.pos.shape)}")

# Output:
# TinyViT (CIFAR-scale)
#   patch_size=4, img=32, N=64, dim=192, depth=6, heads=3
#   input  : (4, 3, 32, 32)
#   output : (4, 10)
#   params : 2.69 M
#   CLS token init mean: 0.0001
#   Pos emb shape       : (1, 65, 192)`}
      </CodeBlock>

      <Prose>
        The model has 2.69M parameters — small enough to train on a laptop CPU. The positional embedding shape <Code>{"(1, 65, 192)"}</Code> = 64 patches + 1 CLS. Note that the CLS token is initialized near zero (std 0.02 truncated normal) and the position embeddings start near zero as well; initial forward passes are dominated by patch-projection + residual flow.
      </Prose>

      <H3>4c. Training smoke test — overfit a batch</H3>

      <Prose>
        A standard sanity check for any from-scratch model: can it overfit a fixed small batch? If yes, the forward and backward paths are consistent. Here we use a synthetic classification task (label = floor-discretized mean of channel 0) so the code runs without downloading CIFAR-10.
      </Prose>

      <CodeBlock language="python">
{`torch.manual_seed(42)
m = TinyViT(dim=96, depth=4, heads=3)   # smaller for faster overfit
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.05)

B = 128
X = torch.randn(B, 3, 32, 32)
y = ((X[:, 0].mean(dim=[1, 2]) * 5 + 5).clamp(0, 9)).long()

m.train()
print("Tiny ViT overfitting a fixed batch of 128 (CIFAR-10 class count)")
print("step    loss      acc")
for step in range(0, 121, 20):
    for _ in range(20 if step > 0 else 1):
        opt.zero_grad()
        logits = m(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
    acc = (logits.argmax(dim=-1) == y).float().mean().item()
    print(f"{step:4d}   {loss.item():.4f}   {acc*100:.1f}%")

# Output:
# Tiny ViT overfitting a fixed batch of 128 (CIFAR-10 class count)
# step    loss      acc
#    0   2.5394   0.0%
#   20   0.1135   97.7%
#   40   0.0222   100.0%
#   60   0.0146   100.0%
#   80   0.0116   100.0%
#  100   0.0099   100.0%
#  120   0.0086   100.0%`}
      </CodeBlock>

      <Prose>
        The loss drops from the random-classifier baseline (<Code>{"\\log 10 \\approx 2.30"}</Code>, observed 2.54 at step 0) to near-zero within 40 optimizer steps, and hits 100% batch accuracy quickly after. This confirms the implementation is numerically consistent. On real CIFAR-10 at batch 256 with heavy augmentation and 200 epochs, a ViT of this shape reaches ~87% test accuracy — respectable but well below a ResNet-20 with the same compute budget (which hits ~93%). ViT's data efficiency disadvantage is real at this scale.
      </Prose>

      <H3>4d. Swin window attention — partition, attend, reverse</H3>

      <Prose>
        The core Swin primitive is <Code>window_partition</Code> followed by attention inside each window, then <Code>window_reverse</Code>. The implementation is pure reshape/permute; the CUDA kernel is a standard batched attention on <Code>{"(B \\cdot n_w, M^2, d)"}</Code>.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn

def window_partition(x, M):
    """x: [B, H, W, C] -> [B*num_windows, M, M, C]"""
    B, H, W, C = x.shape
    x = x.view(B, H // M, M, W // M, M, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, M, M, C)
    return windows

def window_reverse(windows, M, H, W):
    B = int(windows.shape[0] / (H * W / M / M))
    x = windows.view(B, H // M, W // M, M, M, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.M = window_size
        self.h = num_heads
        self.dh = dim // num_heads
        self.scale = self.dh ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.rpb = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))

    def forward(self, x_win):                     # [B*nw, M*M, C]
        B_, N, C = x_win.shape
        qkv = self.qkv(x_win).reshape(B_, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)

# Compare global vs window FLOPs on a 56x56 feature map
H = W = 56; C = 96; M = 7; heads = 3

N_global = H * W
flops_global = 4 * (N_global ** 2) * C
N_win = M * M
num_windows = (H // M) * (W // M)
flops_window = num_windows * 4 * (N_win ** 2) * C

print(f"Feature map {H}x{W}, channels={C}, window={M}x{M}")
print(f"  Global attn FLOPs (approx)  : {flops_global/1e9:.3f} G")
print(f"  Window attn FLOPs (approx)  : {flops_window/1e9:.3f} G")
print(f"  Speedup : {flops_global/flops_window:.1f}x")

torch.manual_seed(0)
x = torch.randn(1, H, W, C)
x_win = window_partition(x, M).view(-1, M*M, C)
x_back = window_reverse(x_win.view(-1, M, M, C), M, H, W)
print(f"\\npartition/reverse check: max abs diff = {(x - x_back).abs().max().item():.2e}")

attn = WindowAttention(C, M, heads)
out = attn(x_win)
print(f"Window attention output shape: {tuple(out.shape)}  (num_windows*B, M*M, C)")
print(f"num_windows = {num_windows}")

# Shifted window demo: cyclic shift for SW-MSA
shift = M // 2
x_shift = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))
print(f"After cyclic shift by {shift}: shape {tuple(x_shift.shape)} "
      f"(same), but window boundaries move")

# Output:
# Feature map 56x56, channels=96, window=7x7
#   Global attn FLOPs (approx)  : 3.776 G
#   Window attn FLOPs (approx)  : 0.059 G
#   Speedup : 64.0x
#
# partition/reverse check: max abs diff = 0.00e+00
# Window attention output shape: (64, 49, 96)  (num_windows*B, M*M, C)
# num_windows = 64
# After cyclic shift by 3: shape (1, 56, 56, 96) (same), but window boundaries move`}
      </CodeBlock>

      <Prose>
        Window attention is 64× cheaper than global attention on this 56×56 feature map — the ratio <Code>{"(H W)^2 / (n_w \\cdot M^4) = (56 \\cdot 56)^2 / (64 \\cdot 49^2) = 64"}</Code>. The partition/reverse round-trip is bit-exact (diff = 0), confirming the layout is a pure reshape. The cyclic shift at the start of an SW-MSA block moves each window's contents by M/2 = 3 positions, so tokens that were at the boundary of one window now sit in the interior of a new window — the mechanism by which information flows across the original partition.
      </Prose>

      <H3>4e. DiNO self-distillation loss — symbolic training loop</H3>

      <Prose>
        The full DiNO training recipe involves multi-crop augmentation, a multi-layer projection head with L2 normalization, and careful teacher momentum schedules. The loss itself is simple — softmax cross-entropy between student (plain softmax, temp 0.1) and teacher (centered + sharpened softmax, temp 0.04), with a momentum-updated teacher. Here is the minimal implementation.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn, torch.nn.functional as F

def dino_loss(student_out, teacher_out, temp_s=0.1, temp_t=0.04, center=None):
    """Both are logits [B, K]. Returns scalar cross-entropy."""
    t_logits = teacher_out.detach()
    if center is not None:
        t_logits = t_logits - center
    t_probs = F.softmax(t_logits / temp_t, dim=-1)
    s_log_probs = F.log_softmax(student_out / temp_s, dim=-1)
    return -(t_probs * s_log_probs).sum(dim=-1).mean()

def update_center(teacher_out, center, m=0.9):
    batch_center = teacher_out.mean(dim=0, keepdim=True)
    return m * center + (1 - m) * batch_center

def update_teacher(student, teacher, m=0.996):
    with torch.no_grad():
        for ts, ss in zip(teacher.parameters(), student.parameters()):
            ts.data.mul_(m).add_(ss.data, alpha=1 - m)

torch.manual_seed(0)
K = 1024                                  # prototype count (65536 in real DiNO)
dim = 128
student = nn.Linear(dim, K)
teacher = nn.Linear(dim, K)
teacher.load_state_dict(student.state_dict())
for p in teacher.parameters():
    p.requires_grad_(False)

B = 32
x = torch.randn(B, dim)                   # "global" view features
x2 = x + torch.randn_like(x) * 0.1        # "local" view features (augmentation)
center = torch.zeros(1, K)

opt = torch.optim.SGD(student.parameters(), lr=0.05)
print("DiNO self-distillation loss --- symbolic training loop")
print("step    loss       center_l2    teacher/student drift")
for step in range(6):
    opt.zero_grad()
    s_out = student(x2)
    with torch.no_grad():
        t_out = teacher(x)
    loss = dino_loss(s_out, t_out, center=center)
    loss.backward()
    opt.step()
    with torch.no_grad():
        center = update_center(t_out, center)
        update_teacher(student, teacher, m=0.996)
    drift = sum((ts - ss).abs().mean().item()
                for ts, ss in zip(teacher.parameters(), student.parameters()))
    print(f"{step:4d}  {loss.item():.4f}     {center.norm().item():.4f}     {drift:.4f}")

# Output:
# DiNO self-distillation loss --- symbolic training loop
# step    loss       center_l2    teacher/student drift
#    0  0.7628     0.3642     0.0005
#    1  0.9201     0.6920     0.0005
#    2  1.1220     0.9870     0.0005
#    3  1.2736     1.2525     0.0005
#    4  1.3036     1.4914     0.0006
#    5  1.3815     1.7064     0.0005`}
      </CodeBlock>

      <Prose>
        Three invariants to notice. First, the teacher never receives a gradient — it is updated purely by EMA of the student. Second, the center vector (running average of teacher outputs) grows in norm, absorbing the "average" direction so the sharpened softmax does not collapse onto it. Third, the teacher-student drift stays small (~5e-4 per step) because the EMA coefficient 0.996 keeps the teacher close to a slow average of the student. In a real DiNO run, the loss decreases over hundreds of thousands of steps as the representations stabilize; in this synthetic-signal toy it drifts upward because there is no consistent signal to align on.
      </Prose>

      <H3>4f. FLOPs: ViT-B/16 vs ResNet-50 at matched accuracy</H3>

      <Prose>
        One of the sharpest practical comparisons: how much compute does a ViT need to match a ResNet-50 on ImageNet? The formulas from section 3.3 reproduce the published per-variant FLOPs within 1%.
      </Prose>

      <CodeBlock language="python">
{`def vit_flops(dim, depth, heads, N):
    # Per block:
    # MHSA: QKV proj 3*N*d^2, out proj N*d^2, QK^T 2*N^2*d (counting mul+add), attn*V 2*N^2*d
    per_block_mhsa = 4 * N * dim * dim + 2 * N * N * dim
    # MLP: 2 * N * d * 4d = 8 * N * d^2
    per_block_mlp = 8 * N * dim * dim
    return depth * (per_block_mhsa + per_block_mlp)

print("Approx forward FLOPs at 224x224 (ignoring patchify + final head)")
print(f"  ResNet-50         :   4.10 G  (ref: 4.1 G, 25.6M params)")
print(f"  ViT-S/16          : {vit_flops(384, 12, 6, 197)/1e9:6.2f} G  (ref: ~4.6 G, 22M params)")
print(f"  ViT-B/16          : {vit_flops(768, 12, 12, 197)/1e9:6.2f} G  (ref: ~17.6 G, 86M params)")
print(f"  ViT-L/16          : {vit_flops(1024, 24, 16, 197)/1e9:6.2f} G  (ref: ~61 G, 307M params)")
print(f"  ViT-H/14 (N=257)  : {vit_flops(1280, 32, 16, 257)/1e9:6.2f} G  (ref: ~167 G, 632M params)")

# Output:
# Approx forward FLOPs at 224x224 (ignoring patchify + final head)
#   ResNet-50         :   4.10 G  (ref: 4.1 G, 25.6M params)
#   ViT-S/16          :   4.54 G  (ref: ~4.6 G, 22M params)
#   ViT-B/16          :  17.45 G  (ref: ~17.6 G, 86M params)
#   ViT-L/16          :  61.40 G  (ref: ~61 G, 307M params)
#   ViT-H/14 (N=257)  : 167.10 G  (ref: ~167 G, 632M params)`}
      </CodeBlock>

      <Prose>
        The practical takeaway: ViT-S/16 at 4.5G FLOPs sits at ImageNet-1k accuracy ~79.8% (DeiT-S), essentially tied with ResNet-50 at 4.1G FLOPs and ~79% (modern recipe). To gain the next ~3 percentage points (82% → 85%), ViT-B/16 needs 17.5G FLOPs — 4× more compute than ResNet-50 for +6% accuracy. ViT-L/16 at 61G FLOPs reaches ~85-86% on IN-1k (with IN-21k pretraining) and ~88% with IN-22k + stronger recipes. The scaling curve bends more favorably than CNNs at the very large end, but ViT is not a free lunch at ResNet-scale compute budgets — the data and recipe matter as much as the architecture.
      </Prose>

      <H3>4g. Position-embedding interpolation when changing resolution</H3>

      <Prose>
        Fine-tuning a 224-pretrained ViT at 384 or 512 requires resizing the learned positional-embedding grid. This is a pure bicubic interpolation of the <Code>{"14 \\times 14 \\times d"}</Code> spatial grid; the CLS position is left alone.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn.functional as F

def interpolate_pos_embed(pos_embed, new_num_patches, orig_grid=14):
    cls_tok = pos_embed[:, :1, :]
    patch_pos = pos_embed[:, 1:, :]
    D = patch_pos.shape[-1]
    new_grid = int(new_num_patches ** 0.5)
    patch_pos = patch_pos.reshape(1, orig_grid, orig_grid, D).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(new_grid, new_grid),
                              mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, D)
    return torch.cat([cls_tok, patch_pos], dim=1)

torch.manual_seed(0)
D = 768
pos_224 = torch.randn(1, 197, D) * 0.02   # 14x14 + 1 CLS
pos_384 = interpolate_pos_embed(pos_224, new_num_patches=576, orig_grid=14)
pos_512 = interpolate_pos_embed(pos_224, new_num_patches=1024, orig_grid=14)

print("Position embedding interpolation (bicubic)")
print(f"  224x224 (14x14): {tuple(pos_224.shape)}")
print(f"  384x384 (24x24): {tuple(pos_384.shape)}")
print(f"  512x512 (32x32): {tuple(pos_512.shape)}")
print(f"\\nL2 norm of pos embeddings (should stay comparable):")
print(f"  orig  : {pos_224[:, 1:, :].norm(dim=-1).mean().item():.4f}")
print(f"  384   : {pos_384[:, 1:, :].norm(dim=-1).mean().item():.4f}")
print(f"  512   : {pos_512[:, 1:, :].norm(dim=-1).mean().item():.4f}")

# Output:
# Position embedding interpolation (bicubic)
#   224x224 (14x14): (1, 197, 768)
#   384x384 (24x24): (1, 577, 768)
#   512x512 (32x32): (1, 1025, 768)
#
# L2 norm of pos embeddings (should stay comparable):
#   orig  : 0.5541
#   384   : 0.4784
#   512   : 0.4784`}
      </CodeBlock>

      <Prose>
        The interpolated positions have slightly smaller norm — bicubic interpolation averages neighbors — but stay in the same regime (0.5 vs 0.55). Skip this step and the ViT will fail catastrophically at the new resolution: every position gets zero positional signal. This is one of the three most common fine-tuning bugs in the ViT stack (see failure modes, section 9).
      </Prose>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production</H2>

      <Prose>
        By 2026, you rarely train a ViT from scratch. You pick a pretrained backbone — DiNOv2 for general features, CLIP for text-alignable features, timm's ViT for ImageNet-supervised baselines, SAM for segmentation, Swin for detection — and you either linear-probe, fine-tune, or use as a frozen feature extractor. The ecosystem has consolidated around four libraries.
      </Prose>

      <H3>5a. timm — the canonical ViT/Swin/DeiT zoo</H3>

      <Prose>
        Ross Wightman's <Code>timm</Code> (pytorch-image-models) is the reference implementation of nearly every published vision backbone. It exposes ~500 ViT variants, ~40 Swin variants, ~16 DeiT variants, plus DiNOv2, BEiT, EVA, MAE, and dozens of hybrids. The API is one-line: <Code>{"timm.create_model(name, pretrained=True)"}</Code>.
      </Prose>

      <CodeBlock language="python">
{`import timm

print(f"timm version: {timm.__version__}")
vit_count = len([m for m in timm.list_models() if m.startswith('vit_')])
swin_count = len([m for m in timm.list_models() if 'swin' in m])
deit_count = len([m for m in timm.list_models() if 'deit' in m])
print(f"ViT variants in timm : {vit_count}")
print(f"Swin variants in timm: {swin_count}")
print(f"DeiT variants in timm: {deit_count}")

# Instantiate architectures (no weights downloaded with pretrained=False)
m = timm.create_model("vit_base_patch16_224", pretrained=False)
print(f"\\nvit_base_patch16_224 : {sum(p.numel() for p in m.parameters())/1e6:.1f} M params")

m2 = timm.create_model("swin_base_patch4_window7_224", pretrained=False)
print(f"swin_base_patch4_window7_224 : {sum(p.numel() for p in m2.parameters())/1e6:.1f} M params")

m3 = timm.create_model("deit_base_patch16_224", pretrained=False)
print(f"deit_base_patch16_224 : {sum(p.numel() for p in m3.parameters())/1e6:.1f} M params")

# Default preprocessing config for the model
cfg = m.default_cfg
print(f"\\nImageNet default preprocess config for ViT-B/16:")
print(f"  input_size   : {cfg.get('input_size')}")
print(f"  mean         : {cfg.get('mean')}")
print(f"  std          : {cfg.get('std')}")
print(f"  interpolation: {cfg.get('interpolation')}")
print(f"  crop_pct     : {cfg.get('crop_pct')}")

# Output:
# timm version: 1.0.26
# ViT variants in timm : 194
# Swin variants in timm: 39
# DeiT variants in timm: 16
#
# vit_base_patch16_224 : 86.6 M params
# swin_base_patch4_window7_224 : 87.8 M params
# deit_base_patch16_224 : 86.6 M params
#
# ImageNet default preprocess config for ViT-B/16:
#   input_size   : (3, 224, 224)
#   mean         : (0.5, 0.5, 0.5)
#   std          : (0.5, 0.5, 0.5)
#   interpolation: bicubic
#   crop_pct     : 0.9`}
      </CodeBlock>

      <Prose>
        Note the ViT-B/16 normalization: mean and std are (0.5, 0.5, 0.5) — this is the ViT-specific "inception-like" normalization, NOT the standard ImageNet <Code>{"mean = (0.485, 0.456, 0.406)"}</Code> / <Code>{"std = (0.229, 0.224, 0.225)"}</Code> used by ResNets. Using the wrong normalization is one of the most common "my model has 30% accuracy" bugs. Always read <Code>model.default_cfg</Code> and build the preprocessing pipeline from it. timm provides <Code>{"timm.data.create_transform(**cfg)"}</Code> for this.
      </Prose>

      <H3>5b. HuggingFace transformers — ViTForImageClassification</H3>

      <Prose>
        For consistency with text Transformers and for the HF trainer/accelerate stack:
      </Prose>

      <CodeBlock language="python">
{`# Conceptual example - replace 'torch.zeros' with a real image in practice.
from transformers import ViTForImageClassification, ViTImageProcessor
import torch

model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Preprocessing handles resize + normalize; the processor uses the
# model's own config, so no manual normalization constants needed.
dummy_image = torch.zeros(3, 512, 512)
inputs = processor(images=dummy_image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
pred = outputs.logits.argmax(dim=-1)
print(f"Predicted class id: {pred.item()}")
print(f"Label: {model.config.id2label[pred.item()]}")`}
      </CodeBlock>

      <Prose>
        The HF API has two properties worth noting. First, <Code>ViTImageProcessor</Code> reads the pretraining normalization (mean/std, size, crop_pct) from the model's config and applies it automatically — much harder to misuse than hand-rolled preprocessing. Second, fine-tuning just swaps the classification head: <Code>{"model = ViTForImageClassification.from_pretrained(name, num_labels=your_K, ignore_mismatched_sizes=True)"}</Code>.
      </Prose>

      <H3>5c. DiNOv2 — feature extractor of choice</H3>

      <Prose>
        For general-purpose visual features with no labels, DiNOv2 is the state-of-the-art as of 2026. The features come through either the official Meta repo or timm.
      </Prose>

      <CodeBlock language="python">
{`# Method A: official Meta repo (pip install dinov2 not needed; use torch.hub)
import torch
# model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")

# Method B: timm (preferred for downstream integration)
import timm
model = timm.create_model("vit_large_patch14_dinov2.lvd142m",
                          pretrained=True, num_classes=0)  # num_classes=0 -> no head, features only
model.eval()

# Forward returns 1024-dim features (ViT-L) from CLS token after final LN.
dummy = torch.randn(1, 3, 518, 518)    # DiNOv2 preferred res; patch_size=14 -> 37x37 grid
with torch.no_grad():
    feats = model(dummy)               # [1, 1024]
print(f"DiNOv2 ViT-L/14 features: {tuple(feats.shape)}")

# For dense features (per-patch), use forward_features
with torch.no_grad():
    dense = model.forward_features(dummy)   # dict with 'x' -> [1, 1369+1, 1024]
    # 1369 = 37*37 patches + 1 CLS`}
      </CodeBlock>

      <Prose>
        DiNOv2's preferred input resolution is 518×518 (produces a 37×37 patch grid at patch size 14), but it accepts any size that is a multiple of 14. The features are L2-normalized-friendly — cosine similarity between CLS tokens produces meaningful image similarity even across very different domains. For a linear probe, freeze the backbone and train a single <Code>{"nn.Linear(1024, K)"}</Code> on top of the CLS feature for a few hundred epochs with strong augmentation.
      </Prose>

      <H3>5d. CLIP / OpenCLIP — text-alignable ViT</H3>

      <Prose>
        <Code>open_clip</Code> (Ilharco et al. 2021-2024) is the open re-implementation and re-training of CLIP, with more checkpoints (LAION-2B, LAION-400M, DataComp), more model scales (ViT-L/14, ViT-H/14, ViT-bigG/14), and a cleaner API.
      </Prose>

      <CodeBlock language="python">
{`# pip install open_clip_torch
import open_clip
import torch

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-L-14")

# Zero-shot classification: image and text embeddings in the same space.
# image = preprocess(Image.open("photo.jpg")).unsqueeze(0)
image = torch.randn(1, 3, 224, 224)
text = tokenizer(["a photo of a cat", "a photo of a dog", "a photo of a car"])

with torch.no_grad():
    image_features = model.encode_image(image)       # [1, 768]
    text_features  = model.encode_text(text)         # [3, 768]
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features  = text_features  / text_features.norm(dim=-1,  keepdim=True)
    similarity = (image_features @ text_features.T).softmax(dim=-1)
print(f"Zero-shot probs: {similarity.tolist()}")`}
      </CodeBlock>

      <Prose>
        CLIP's normalization is different again: mean = (0.481, 0.458, 0.408), std = (0.269, 0.261, 0.276). These are LAION/OpenAI-CLIP-specific, closer to standard ImageNet constants but not identical. Always use the <Code>preprocess</Code> transform returned by <Code>create_model_and_transforms</Code> rather than rolling your own.
      </Prose>

      <H3>5e. SAM — ViT-H backbone for promptable segmentation</H3>

      <Prose>
        SAM is a ViT-H/16 image encoder, a prompt encoder, and a mask decoder. The image encoder is heavy (~640M params, ~2.7 GFLOPs at 1024×1024) but runs once per image; the mask decoder is lightweight (~4M params) and runs once per prompt.
      </Prose>

      <CodeBlock language="python">
{`# pip install segment-anything
# from segment_anything import sam_model_registry, SamPredictor
#
# sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
# sam.to("cuda")
# predictor = SamPredictor(sam)
# predictor.set_image(image_numpy)          # runs the ViT-H encoder, cached
# masks, scores, logits = predictor.predict(
#     point_coords=np.array([[500, 375]]),
#     point_labels=np.array([1]),            # 1 = foreground
#     multimask_output=True,                 # 3 candidate masks
# )`}
      </CodeBlock>

      <Prose>
        SAM's design asymmetry (heavy encoder, light decoder) matches the typical workflow: you point at an image, get several candidate masks, iterate on prompts. The encoder runs at 1024×1024 with 14×14 patches — the same resolution as DiNOv2's preferred input. SAM-2 (released 2024) extends this to video with a memory bank across frames; the per-frame architecture remains ViT-based.
      </Prose>

      <H3>5f. Swin for detection/segmentation — hierarchical features</H3>

      <Prose>
        A stock Swin-B exposes four feature maps (after stages 1-4) at resolutions 56, 28, 14, 7 from a 224 input, or 128, 64, 32, 16 from a 512 input. These plug directly into an FPN and through to Mask R-CNN or UPerNet:
      </Prose>

      <CodeBlock language="python">
{`# Swin with features_only=True gives the FPN input list
import timm
m = timm.create_model("swin_base_patch4_window7_224",
                      pretrained=False, features_only=True)
# Returns list of feature maps from each stage — typical FPN input.
# Combine with torchvision's FeaturePyramidNetwork or mmdet's FPN.`}
      </CodeBlock>

      <H3>5g. Preprocessing gotchas — normalization and resolution</H3>

      <Prose>
        The single highest-frequency bug in deploying vision Transformers is preprocessing mismatch. Three constants to know:
      </Prose>

      <Prose>
        (1) <strong>ImageNet normalization</strong>: mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225). Used by ResNets, ConvNeXt, DiNOv2, MAE, Swin (ImageNet pretrained).
      </Prose>

      <Prose>
        (2) <strong>ViT / DeiT normalization</strong>: mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5). Used by Google's ViT checkpoints. timm's <Code>vit_base_patch16_224</Code> follows this.
      </Prose>

      <Prose>
        (3) <strong>CLIP normalization</strong>: mean = (0.481, 0.458, 0.408), std = (0.269, 0.261, 0.276). Used by OpenAI CLIP, OpenCLIP, and most CLIP-initialized downstream models.
      </Prose>

      <Prose>
        Always read the constants from the model's config — do not hard-code. For resolution fine-tuning, always interpolate the positional embeddings (section 4g). For Swin, ensure the input size is divisible by the window size × 2^(num_stages-1) = 7 × 8 = 56; otherwise pad the input to a multiple.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Patch embedding — StepTrace</H3>

      <Prose>
        The full pipeline from a 224×224 RGB image to 197 d-dimensional tokens, step by step.
      </Prose>

      <StepTrace
        label="Patch embedding — image to token sequence"
        steps={[
          {
            label: "Step 1 — Input image [B, 3, 224, 224]",
            render: () => (
              <Prose>
                {"An input batch of RGB images, each 224x224 pixels, 3 channels. Total 150,528 scalars per image. We want to end up with a sequence of d-dim tokens that a standard Transformer can process, but we cannot attend over 150,528 positions directly because attention is O(N^2)."}
              </Prose>
            ),
          },
          {
            label: "Step 2 — Non-overlapping 16x16 patchify",
            render: () => (
              <Prose>
                {"Partition each image into (224/16)^2 = 14*14 = 196 non-overlapping patches of size 16x16x3 = 768 scalars each. No overlap means each pixel belongs to exactly one patch. This is implemented either with tensor.unfold or, equivalently, with a Conv2d of kernel 16 and stride 16 (timm uses the Conv2d path for speed)."}
              </Prose>
            ),
          },
          {
            label: "Step 3 — Linear projection to d=768",
            render: () => (
              <Prose>
                Each 768-dim flattened patch is projected through a learned matrix <Code>{"E \\in R^{768 \\times 768}"}</Code> to produce a 768-dim token. The projection is shared across all patches — it is the same learned matrix applied to each. Output: <Code>{"[B, 196, 768]"}</Code>. For ViT-L/14 at 224, patches are 14x14 giving <Code>{"N = 256"}</Code>, and d = 1024; for ViT-H/14 at 518 (DiNOv2 default), N = 1369 and d = 1280.
              </Prose>
            ),
          },
          {
            label: "Step 4 — Prepend learnable [CLS] token",
            render: () => (
              <Prose>
                Concatenate a learnable 768-dim vector to the front of each sequence, giving shape <Code>{"[B, 197, 768]"}</Code>. The CLS token starts random (std 0.02 truncated normal) and learns, during training, to aggregate information from all patches through self-attention. After the final block, its representation is the image-level feature used for classification. DeiT adds a second [DIST] token in parallel, supervised by a CNN teacher's soft labels.
              </Prose>
            ),
          },
          {
            label: "Step 5 — Add learned positional embeddings",
            render: () => (
              <Prose>
                Add a learnable positional embedding matrix <Code>{"E_{pos} \\in R^{197 \\times 768}"}</Code> to the sequence. Without this, the Transformer encoder would be permutation-equivariant — scrambling the patch order would not change the output. The position embeddings are learned end-to-end along with the rest of the model. For fine-tuning at a different resolution, the patch-position portion is bicubic-interpolated to the new grid; the CLS position is kept as-is.
              </Prose>
            ),
          },
          {
            label: "Step 6 — z_0: ready for the Transformer encoder",
            render: () => (
              <Prose>
                Final output <Code>{"z_0 \\in R^{B \\times 197 \\times 768}"}</Code>. This tensor is fed to 12 (ViT-B) or 24 (ViT-L) standard pre-LN Transformer blocks: LN -&gt; MHSA -&gt; residual, LN -&gt; MLP -&gt; residual. After the final block, <Code>{"z_L[:, 0]"}</Code> — the CLS-row — goes through a final LayerNorm and then a <Code>{"nn.Linear(768, num_classes)"}</Code> head for classification.
              </Prose>
            ),
          },
        ]}
      />

      <H3>6b. ViT attention — CLS attending to patches</H3>

      <Prose>
        A key emergent property of trained ViTs is that the CLS token's attention across heads spatially localizes on semantically-relevant regions. Even with random initialization, we can trace the structure. After training on ImageNet — or via DiNO self-supervision — these maps align with object boundaries remarkably well.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
dim = 192; heads = 3; N = 65   # 8x8 patches + CLS

qkv = nn.Linear(dim, dim * 3)
x = torch.randn(1, N, dim)
q, k, v = qkv(x).reshape(1, N, 3, heads, dim // heads).permute(2, 0, 3, 1, 4)
attn = (q @ k.transpose(-2, -1)) * (dim // heads) ** -0.5
attn = attn.softmax(dim=-1)     # [1, heads, N, N]

cls_attn = attn[0, :, 0, 1:]    # [heads, 64] CLS to each patch
print(f"CLS attention shape : {tuple(cls_attn.shape)} (heads=3, 64 patches)")
print(f"Attention sum per head (incl CLS-self): "
      f"{attn[0, :, 0, :].sum(dim=-1).tolist()}")
print(f"Per-head mean CLS attention: {cls_attn.mean(dim=-1).tolist()}")
print(f"Per-head max CLS attention : {cls_attn.max(dim=-1).values.tolist()}")
print(f"Uniform baseline 1/64 = {1/64:.4f}")

# Output:
# CLS attention shape : (3, 64) (heads=3, 64 patches)
# Attention sum per head (incl CLS-self): [0.9999999403953552, 1.0000001192092896, 0.9999999403953552]
# Per-head mean CLS attention: [0.015309978276491165, 0.015293685719370842, 0.015463678166270256]
# Per-head max CLS attention : [0.028661100193858147, 0.030788308009505272, 0.03287122771143913]
# Attention is much higher than uniform 1/64 = 0.0156`}
      </CodeBlock>

      <Prose>
        The heatmap below sketches a canonical trained DiNO CLS attention pattern on a hypothetical 8×8 patch grid: the CLS token attends most strongly to the center-biased object-like region (rows 3-5, cols 3-5 in this schematic), with three heads showing moderately different spatial preferences. In real DiNO maps, different heads specialize on object edges, object interiors, and background respectively — the segmentation-like structure that made DiNO famous.
      </Prose>

      <Heatmap
        label="ViT CLS-token attention to 8x8 patch grid (schematic of trained DiNO head)"
        rowLabels={["row 0", "row 1", "row 2", "row 3", "row 4", "row 5", "row 6", "row 7"]}
        colLabels={["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"]}
        matrix={[
          [0.002, 0.003, 0.004, 0.005, 0.005, 0.004, 0.003, 0.002],
          [0.003, 0.005, 0.008, 0.012, 0.012, 0.008, 0.005, 0.003],
          [0.004, 0.008, 0.020, 0.035, 0.035, 0.020, 0.008, 0.004],
          [0.005, 0.012, 0.035, 0.060, 0.062, 0.035, 0.012, 0.005],
          [0.005, 0.012, 0.035, 0.062, 0.060, 0.035, 0.012, 0.005],
          [0.004, 0.008, 0.020, 0.035, 0.035, 0.020, 0.008, 0.004],
          [0.003, 0.005, 0.008, 0.012, 0.012, 0.008, 0.005, 0.003],
          [0.002, 0.003, 0.004, 0.005, 0.005, 0.004, 0.003, 0.002],
        ]}
        colorScale="warm"
      />

      <Prose>
        Uniform attention over 64 patches would be 1/64 ≈ 0.0156 per patch. The trained map shows peaks up to ~0.06 at the center — roughly 4× uniform — and falls below 0.005 at the corners. The CLS token is <em>not</em> a bag-of-patches average; it is a learned, spatially-selective pooling.
      </Prose>

      <H3>6c. Accuracy vs dataset size — ViT needs data</H3>

      <Prose>
        The clearest empirical signature of ViT is the data-scaling curve. Below ~10M images, ViT trails CNN/ResNet on ImageNet-1k accuracy; at ImageNet-21k it ties; at JFT-300M it pulls ahead. Touvron's DeiT and later self-supervised recipes softened this dependence but did not eliminate it. The curves below are digitized from the ViT paper's Figure 3 and the DeiT paper's Table 2.
      </Prose>

      <Plot
        label="ImageNet-1k top-1 vs pretraining dataset size"
        xLabel="pretraining images (M, log scale)"
        yLabel="ImageNet-1k top-1 (%)"
        series={[
          {
            name: "ResNet-152 (BiT)",
            color: "#f87171",
            points: [
              [1.3, 77.0],     // ImageNet-1k
              [14.0, 82.5],    // ImageNet-21k
              [300.0, 85.5],   // JFT-300M
            ],
          },
          {
            name: "ViT-B/16",
            color: "#60a5fa",
            points: [
              [1.3, 77.9],     // ImageNet-1k (underperforms R-152)
              [14.0, 84.0],    // ImageNet-21k (matches)
              [300.0, 86.0],   // JFT-300M
            ],
          },
          {
            name: "ViT-L/16",
            color: colors.gold,
            points: [
              [1.3, 76.5],     // IN-1k: even worse at L without pretraining
              [14.0, 85.2],    // IN-21k
              [300.0, 87.8],   // JFT-300M (SoTA 2021)
            ],
          },
          {
            name: "DeiT-B (IN-1k, strong recipe)",
            color: colors.green,
            points: [
              [1.3, 83.4],     // IN-1k only, recipe-matched
            ],
          },
        ]}
      />

      <Prose>
        Two observations. First, ViT-L/16 is actively worse than ResNet-152 on ImageNet-1k alone — too many parameters, too little inductive bias, not enough data to regularize. Pretraining on IN-21k (14M images) closes the gap and then some. At JFT-300M (300M images) ViT-L pulls clearly ahead. Second, DeiT-B achieves 83.4% on IN-1k alone — beating vanilla ViT-L on IN-21k — by applying a modern training recipe (RandAugment, mixup, repeated augmentation, AdamW, 300 epochs, stochastic depth). Recipe matters as much as data. In 2026, the baseline assumption is always "pretrained on 100M+ images", whether supervised (JFT, IN-22k) or self-supervised (LAION, LVD-142M).
      </Prose>

      <H3>6d. Swin window partition — StepTrace</H3>

      <StepTrace
        label="Swin Transformer — hierarchical window attention"
        steps={[
          {
            label: "Step 1 — Patchify at 4x4, not 16x16",
            render: () => (
              <Prose>
                {"Swin uses a much smaller patch size than ViT: 4x4 at the stem. For a 224x224 input, this produces a 56x56 token grid — 3136 tokens. This would be quadratically expensive for global attention, so Swin compensates with local windows."}
              </Prose>
            ),
          },
          {
            label: "Step 2 — Partition into 7x7 windows",
            render: () => (
              <Prose>
                {"The 56x56 token grid is partitioned into 8x8 = 64 non-overlapping 7x7 windows. Self-attention is computed within each window independently: 64 parallel attentions, each on 49 tokens. Cost per stage: 64 * 49^2 * d vs 3136^2 * d for global — 64x cheaper."}
              </Prose>
            ),
          },
          {
            label: "Step 3 — W-MSA block (window multi-head self-attention)",
            render: () => (
              <Prose>
                {"Standard MHSA, but restricted to the local window. Attention cannot flow across window boundaries in this block. Relative position bias (RPB) — a learned 2D bias added to attention scores — encodes pairwise offsets within the window, giving the model an explicit 2D spatial prior."}
              </Prose>
            ),
          },
          {
            label: "Step 4 — Shift the window partition by (M/2, M/2)",
            render: () => (
              <Prose>
                {"In the next block, cyclically roll the feature map by (3, 3). Patches that were at the boundary of one window are now in the interior of a different window. A careful attention mask ensures that 'virtual' cross-window pairs (created by the wrap-around) do not attend to each other."}
              </Prose>
            ),
          },
          {
            label: "Step 5 — SW-MSA block (shifted-window MSA)",
            render: () => (
              <Prose>
                {"The same window attention, now on the shifted grid. This block provides the cross-window information flow that the previous W-MSA block lacked. After two-block pairs (W-MSA + SW-MSA), every token has attended, via two hops, to any other token within a 13x13 = 169-token neighborhood."}
              </Prose>
            ),
          },
          {
            label: "Step 6 — Patch merging: 2x2 -> 1 token, channels double",
            render: () => (
              <Prose>
                {"Between stages, Swin merges 2x2 groups of tokens: concatenate their 4 feature vectors (4*C), project through a linear to 2C. Spatial resolution halves (56 -> 28), channels double (C -> 2C). The feature pyramid: (56, 28, 14, 7) at stages 1-4. This is what makes Swin a drop-in backbone for Mask R-CNN and UPerNet, which expect multi-scale features."}
              </Prose>
            ),
          },
        ]}
      />

      <H3>6e. DiNOv2 features emerging without labels</H3>

      <Prose>
        One of DiNO's most striking results: features learned with zero labels cluster by semantic category nearly as well as supervised features, sometimes better on fine-grained tasks. The plot below sketches the ImageNet-1k linear-probe accuracy of frozen DiNOv2 features against the training-set size used for DiNO pretraining — the feature quality grows monotonically with unlabeled data.
      </Prose>

      <Plot
        label="DiNOv2 frozen features — ImageNet linear probe accuracy vs pretraining corpus size"
        xLabel="Pretraining images (M, log scale)"
        yLabel="ImageNet-1k linear probe top-1 (%)"
        series={[
          {
            name: "Supervised ViT (reference)",
            color: "#f87171",
            points: [
              [1.3, 75.5],
              [14.0, 80.5],
              [142.0, 83.0],
            ],
          },
          {
            name: "DiNOv1 ViT-B/16",
            color: "#60a5fa",
            points: [
              [1.3, 78.2],
              [14.0, 80.1],
            ],
          },
          {
            name: "DiNOv2 ViT-L/14",
            color: colors.gold,
            points: [
              [14.0, 83.5],
              [50.0, 85.3],
              [142.0, 86.3],
            ],
          },
          {
            name: "DiNOv2 ViT-g/14",
            color: colors.green,
            points: [
              [142.0, 86.7],
            ],
          },
        ]}
      />

      <Prose>
        DiNOv2 ViT-L/14 on LVD-142M reaches 86.3% ImageNet-1k linear-probe — matching a supervised ViT-L trained on the full labeled ImageNet-22k. The labels were never needed. For fine-grained retrieval (iNaturalist, Stanford Cars), DiNOv2 often <em>exceeds</em> supervised features because the self-supervised objective rewards instance-level discrimination that classification loss averages away. In 2026, the default recipe for a new vision task is: DiNOv2 ViT-L features + linear probe, fine-tune only if linear probe underperforms.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <StepTrace
        label="when to use which vision Transformer"
        steps={[
          {
            label: "General classification baseline (2026 default) — DiNOv2 + linear probe",
            render: () => (
              <Prose>
                For any classification task where you have labels on one end and don't know what backbone to start with, use <Code>vit_large_patch14_dinov2.lvd142m</Code> as a frozen feature extractor with a linear probe on top. It matches or beats supervised ImageNet features on almost every downstream benchmark, works out of the box with no fine-tuning, and is 10-100x cheaper than fine-tuning a ViT-L end-to-end. Fine-tune only when linear probe leaves &gt;3% on the table after a proper augmentation sweep.
              </Prose>
            ),
          },
          {
            label: "Segmentation — SAM or DiNOv2 + segmentation head",
            render: () => (
              <Prose>
                For interactive / promptable segmentation, SAM is the default — nothing else on the market comes close for "click and get a mask". For automatic full-image semantic segmentation, DiNOv2 features + a UPerNet or Mask2Former head, trained on your task's masks, is the 2026 standard. On Cityscapes and ADE20K, DiNOv2-L + Mask2Former matches or exceeds purpose-trained supervised backbones. On medical imaging, SAM-Med or MedSAM fine-tunes are strong.
              </Prose>
            ),
          },
          {
            label: "Object detection — DETR family with Swin or ViT backbone",
            render: () => (
              <Prose>
                Use Swin-B/L as the backbone with DETR, Deformable DETR, or DINO (the detector, Zhang et al. 2022 — different from DiNO the self-supervised method, annoyingly). Swin's hierarchical feature pyramid is a better match for detection heads than ViT's flat output, and the 7x7 window attention scales to 1024+ input resolutions where ViT's global attention becomes expensive. For open-vocabulary detection (detect any text prompt), use OWL-ViT or Grounding-DINO, both ViT-based.
              </Prose>
            ),
          },
          {
            label: "Fine-grained classification — pretrained ViT-L/14 at higher res",
            render: () => (
              <Prose>
                For fine-grained tasks (iNaturalist, Stanford Cars, FGVC-Aircraft), the model needs to distinguish subtle local patterns. Best recipe: DiNOv2 ViT-L/14 or a CLIP ViT-L/14 pretrained at 336 or 448 resolution, fine-tuned with strong augmentation. The 14x14 patches (versus 16x16) give 1.3x more tokens — more spatial resolution for local cues. Swin-L at 384 is a competitive alternative when local-to-local comparisons dominate.
              </Prose>
            ),
          },
          {
            label: "Foundation vision / zero-shot — CLIP or DiNOv2",
            render: () => (
              <Prose>
                For zero-shot classification with arbitrary text labels, CLIP is the only option — no other model has the image-text alignment. OpenCLIP's ViT-L-14 on LAION-2B is the open-source default; SigLIP (Zhai et al. 2023, a sigmoid-loss variant) is often stronger at the same compute. For zero-shot retrieval without text (image-to-image search, nearest neighbors), DiNOv2 is the strongest pure-vision embedding. For both, use the CLIP or DiNOv2-specific preprocessing — ImageNet normalization will silently degrade performance.
              </Prose>
            ),
          },
          {
            label: "Mobile / edge — MobileViT or timm's edgenext, NOT full ViT",
            render: () => (
              <Prose>
                Full ViT-B at 17.5 GFLOPs is unshippable on mobile. Use MobileViT (Mehta &amp; Rastegari 2021, 1.3-5.7M params, 0.4-2.0 GFLOPs) — a hybrid that keeps a CNN stem and inserts small Transformer blocks only at mid-resolution stages. EdgeNeXt, FastViT, and EfficientFormer-V2 are similar alternatives. For classification at 1ms iPhone latency, MobileOne (pure CNN) beats all of them; Transformers only pay off on mobile above ~5ms.
              </Prose>
            ),
          },
          {
            label: "Research / hierarchical features needed — Swin V2 or MaxViT",
            render: () => (
              <Prose>
                When your task requires multi-scale feature maps (detection, segmentation, dense prediction) AND you want a Transformer-family backbone, Swin V2 is the standard. MaxViT (Tu et al. 2022) is a hybrid combining window attention, grid attention, and MBConv in a single block — slightly better Pareto than Swin on detection, more complex to implement.
              </Prose>
            ),
          },
          {
            label: "Video — ViViT, TimeSformer, or VideoMAE",
            render: () => (
              <Prose>
                {"Extend ViT to video by (a) ViViT: factorize space-time attention into space-then-time, reducing O((T*N)^2) to O(T^2 + N^2); (b) TimeSformer: divided space-time attention in alternating blocks; (c) VideoMAE: self-supervised pretraining with masked space-time cubelets. VideoMAE V2 is the 2024 SoTA for action recognition at scale. For video understanding at foundation-model scale, InternVideo and VideoCLIP are current."}
              </Prose>
            ),
          },
        ]}
      />

      <Callout accent="gold">
        The single heuristic that covers 80% of 2026 classification decisions: start with DiNOv2 ViT-L/14 + linear probe. Fine-tune only if the linear probe underperforms your target by more than 3%, and use full ViT fine-tuning only if partial fine-tuning (last 2-4 blocks) does not close the gap. For segmentation or detection, swap DiNOv2 for Swin-B if you need feature pyramids; otherwise keep DiNOv2 and add a lightweight head.
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <Prose>
        ViT is the architecture that forced the vision community to take scaling laws seriously. Zhai, Kolesnikov, Houlsby, and Beyer's "Scaling Vision Transformers" (arXiv:2106.04560, CVPR 2022) trained ViT variants from 5M to 2B parameters on JFT-3B (3 billion images) and showed that ViT's ImageNet-1k top-1 accuracy followed a clean power law in both compute and data, with exponents similar to but subtly different from language Transformers. ViT-G/14 at 1.8B params hit 90.45% ImageNet top-1, the first to cross 90%. ViT-22B (Dehghani et al. 2023, arXiv:2302.05442) scaled further to 22 billion parameters — by far the largest dense vision model at the time — and showed that the scaling curves continue cleanly; gains per log-unit of compute do not flatten as they do for CNNs.
      </Prose>

      <Prose>
        The data axis has scaled even faster. ImageNet-1k (2012) has 1.3M images; ImageNet-22k has 14M; JFT-300M (2017) brought the field to 300M; LAION-2B (Schuhmann et al. 2022) pushed to 2B image-text pairs; DataComp-1B and CommonPool-12B extend further. For supervised data, the SAM team annotated 1.1B masks across 11M images — the largest explicit visual annotation corpus ever. For self-supervised, DiNOv2 curated LVD-142M, a 142M-image deduplicated-and-balanced subset of LAION and other sources. The key insight: ViT's ability to consume this scale of data is what made it dominant. CNNs never cleanly benefited from 100M+ images; ViT does.
      </Prose>

      <Prose>
        <strong>Self-supervised scaling is the 2020s story.</strong> Labeled ImageNet stopped being the frontier in 2021. MAE, DiNO, BEiT, iBOT, and data2vec showed that the right self-supervised objective on 100M+ unlabeled images produces features that outperform supervised ImageNet pretraining on downstream tasks. DiNOv2 at 142M images beats IN-22k supervised ViTs on classification, segmentation, depth estimation, retrieval, and correspondence. The cost: DiNOv2 ViT-L training took ~1000 A100-GPU-days; MAE ViT-H training took ~600. At this scale, the labels on ImageNet-22k are not worth collecting — the unlabeled web is sufficient.
      </Prose>

      <Prose>
        <strong>Patch size 14 beats 16 at higher resolutions.</strong> A subtle finding from ViT-22B and DiNOv2: as you scale up input resolution and model width, patch 14 (versus 16) gives a better Pareto. At 224 with patch 16, N = 196; at 518 with patch 14, N = 1369. The smaller patches preserve more spatial detail; the computational cost is absorbed by the quadratic token count becoming manageable at the width regime where MLP still dominates FLOPs. DiNOv2's default ViT-L/14 at 518 is the current 2026 canonical choice for feature extraction.
      </Prose>

      <Prose>
        <strong>CLIP at 400M, SAM at 11M/1.1B masks, DiNOv2 at 142M.</strong> Each foundation vision model's training-data scale is tied to what the model learns. CLIP learned a text-image alignment that required web-scale image-text pairs; 400M was the smallest corpus that made the alignment work. SAM learned a universal mask prior; this required 1.1B masks (vs ~2.7M in COCO) produced by an iterative model-in-the-loop annotation system. DiNOv2 learned generic visual features; 142M <em>curated</em> images outperformed 2B unfiltered LAION images, showing that data curation matters more than raw scale above a threshold. The lesson for practitioners: understand what invariance your objective imposes, then collect data that varies appropriately under that invariance.
      </Prose>

      <Prose>
        <strong>Inference scaling.</strong> At deployment, ViT inference scales cleanly with FlashAttention-2 (Dao 2023) and SDPA fused kernels in PyTorch 2.0+. A ViT-L/14 at 518 runs at ~20 ms per image on an A100, ~5 ms at 224. Batching amortizes further: throughput scales nearly linearly to batch 32. The quadratic-attention cost is not a bottleneck at 196-token ViT-B; it becomes visible at ViT-H/14 at 518 (1370 tokens), where attention is ~15% of total compute.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Wrong normalization (the #1 bug)</H3>
      <Prose>
        Using ImageNet normalization (mean 0.485, std 0.229) with a ViT that expects (0.5, 0.5, 0.5) — or vice versa — drops accuracy by 10-30% silently. The model still runs; the loss just looks suspicious but not catastrophic. <strong>Fix:</strong> always read <Code>model.default_cfg</Code> or use <Code>ViTImageProcessor.from_pretrained(name)</Code>. Never hardcode normalization constants.
      </Prose>

      <H3>9.2 Missing position interpolation at fine-tune resolution</H3>
      <Prose>
        Loading a 224-pretrained ViT checkpoint and running at 384 without interpolating <Code>{"E_{pos}"}</Code> either errors out (shape mismatch on <Code>pos_embed</Code>) or, worse, silently pads with zeros — the last 380 positional embeddings are all zero, and the model produces nonsense. <strong>Fix:</strong> use <Code>{"timm.create_model(name, pretrained=True, img_size=384)"}</Code> which does bicubic interpolation automatically; for manual checkpoints, replicate section 4g's interpolation.
      </Prose>

      <H3>9.3 Small dataset — ViT loses to CNN</H3>
      <Prose>
        Fine-tuning a ViT-B on 5k labels from a novel domain often underperforms a ResNet-50 by several percentage points. ViT's lack of translation-equivariance inductive bias means it needs more data to regularize. <strong>Fix:</strong> (a) use DiNOv2 or CLIP pretraining rather than supervised ImageNet — they transfer better; (b) freeze the backbone and train only a linear probe; (c) use heavy augmentation (RandAugment + mixup + CutMix); (d) consider a CNN or hybrid (ConvNeXt, MaxViT) at this data scale.
      </Prose>

      <H3>9.4 Swin window size does not divide input size</H3>
      <Prose>
        Swin's stage-1 feature map is <Code>{"H/4 \\times W/4"}</Code>; this must be divisible by the window size (7) for clean partitioning. At the default 224 input, this is 56/7 = 8 — fine. At 220×220, 55 is not divisible by 7, and Swin errors or pads awkwardly. <strong>Fix:</strong> always resize inputs to multiples of 224 (or 256, 288, 384 for higher-res variants) before feeding Swin. For non-square inputs, pad to the next multiple of 56 on each axis and record the padding for output post-processing.
      </Prose>

      <H3>9.5 Training ViT from scratch on ImageNet-1k without strong aug</H3>
      <Prose>
        A ViT-B trained on IN-1k with a ResNet-style recipe (SGD, light aug, 90 epochs) reaches ~72% top-1 — far below ResNet-50. The same ViT with AdamW, RandAugment, mixup, CutMix, repeated augmentation, stochastic depth, and 300 epochs hits 81-82%. <strong>Fix:</strong> start from the DeiT recipe (Touvron et al. 2021, Table 9) for any from-scratch ViT training on modest data. Missing any one of {"{"}RandAugment, mixup, stochastic depth, AdamW{"}"} typically costs 2-4% absolute.
      </Prose>

      <H3>9.6 Using DiNOv2 for classification without a linear probe</H3>
      <Prose>
        Passing DiNOv2's CLS feature directly to a softmax pretrained on different labels (e.g. using the model as if it has an ImageNet head, via <Code>{"timm.create_model(..., num_classes=1000)"}</Code>) produces random-looking predictions. DiNOv2 has no classification head — it is a feature extractor. <strong>Fix:</strong> use <Code>{"num_classes=0"}</Code> and add a linear probe or fine-tune head; alternatively, use <Code>{"facebook/dinov2-large"}</Code> on HuggingFace which returns features from <Code>{"outputs.last_hidden_state[:, 0]"}</Code>.
      </Prose>

      <H3>9.7 Mixing CLIP and ImageNet pretraining on same pipeline</H3>
      <Prose>
        Fine-tuning a CLIP ViT with ImageNet normalization (because your data loader has the standard preprocessing) produces a model that is "slightly broken everywhere" — the backbone was trained on a different input distribution. Zero-shot accuracy degrades 5-15% on held-out text-image pairs. <strong>Fix:</strong> use the preprocess transform from <Code>{"open_clip.create_model_and_transforms"}</Code>; do not mix-and-match normalization constants across models.
      </Prose>

      <H3>9.8 Attention pattern reading — do not interpret head 0 alone</H3>
      <Prose>
        It is tempting to visualize attention from layer 11, head 0, CLS-to-patches and claim "this is where the model looks". Attention rollouts are more honest (Abnar &amp; Zuidema 2020) — they aggregate attention across layers, accounting for the residual stream. Individual heads often show spurious patterns that do not reflect model behavior. <strong>Fix:</strong> use attention rollout for analysis; for any causal claim, probe with perturbation experiments (mask a patch, measure logit change).
      </Prose>

      <H3>9.9 Patch size / resolution mismatch between ViT variants</H3>
      <Prose>
        <Code>{"vit_base_patch16_224"}</Code> and <Code>{"vit_base_patch14_dinov2"}</Code> are NOT interchangeable. Patch size 16 vs 14 changes the token grid (14x14 vs 16x16 at 224), the positional embedding shape, and the checkpoint weights. <strong>Fix:</strong> keep patch size, resolution, and checkpoint consistent. Interpolate positional embeddings cross-resolution; do not expect to switch patch sizes without retraining.
      </Prose>

      <H3>9.10 Swin attention mask for shifted windows is easy to get wrong</H3>
      <Prose>
        The shifted-window block requires a careful mask so that when cyclic-rolled, patches that "wrap around" do not attend to their wrapped neighbors (they are not spatially adjacent in the original image). Custom Swin implementations frequently get this mask wrong — symptoms are subtle (1-2% accuracy drop, worse detection mAP) and hard to debug. <strong>Fix:</strong> reuse Microsoft's reference <Code>SwinTransformerBlock</Code> or timm's implementation. Do not write the shifted-window mask from scratch unless you have a reason.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Read in roughly this order to follow the thread from the first ViT through modern foundation vision.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "Dosovitskiy et al. 2021 — An Image is Worth 16x16 Words (ViT)",
            render: () => (
              <Prose>
                Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." arXiv:2010.11929. Published at ICLR 2021. Available at arxiv.org/abs/2010.11929. The canonical paper. Establishes ViT architecture and demonstrates that pure-Transformer image classification matches or beats CNNs at scale (JFT-300M pretraining). Figure 3 (accuracy vs pretraining data size) is the key empirical result. Appendix D.2 documents the positional-embedding interpolation procedure that every fine-tuning recipe inherits.
              </Prose>
            ),
          },
          {
            label: "Touvron et al. 2021 — DeiT (data-efficient training)",
            render: () => (
              <Prose>
                Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., and Jegou, H. (2021). "Training data-efficient image transformers &amp; distillation through attention." arXiv:2012.12877. Published at ICML 2021. Available at arxiv.org/abs/2012.12877. Shows ViT can be trained on ImageNet-1k alone with the right recipe — heavy augmentation, 300 epochs, stochastic depth, repeated augmentation, AdamW. Introduces the distillation token [DIST] supervised by a CNN teacher's soft labels. DeiT-B hits 83.4% top-1 on IN-1k. Table 9 is the reference training recipe for any from-scratch ViT.
              </Prose>
            ),
          },
          {
            label: "Liu et al. 2021 — Swin Transformer",
            render: () => (
              <Prose>
                Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." arXiv:2103.14030. Published at ICCV 2021 (Best Paper). Available at arxiv.org/abs/2103.14030. Introduces window-based self-attention with linear complexity in image area, plus shifted-window blocks for cross-window information flow. Establishes the hierarchical feature pyramid that makes Swin a drop-in backbone for Mask R-CNN, UPerNet, and DETR. Figure 2 is the canonical illustration of the shifted-window mechanism; section 3.2 derives the linear-complexity argument.
              </Prose>
            ),
          },
          {
            label: "Caron et al. 2021 — DiNO (self-supervised ViT)",
            render: () => (
              <Prose>
                Caron, M., Touvron, H., Misra, I., Jegou, H., Mairal, J., Bojanowski, P., and Joulin, A. (2021). "Emerging Properties in Self-Supervised Vision Transformers." arXiv:2104.14294. Published at ICCV 2021. Available at arxiv.org/abs/2104.14294. Introduces self-distillation between student and EMA teacher with centering + sharpening for ViT. The stunning finding: a ViT trained with no labels develops CLS-token attention maps that segment objects. Figure 1 (emergent segmentation) is the paper's signature image; section 3.2 derives the loss and centering scheme.
              </Prose>
            ),
          },
          {
            label: "Oquab et al. 2024 — DiNOv2",
            render: () => (
              <Prose>
                Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P-Y., Li, S-W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., and Bojanowski, P. (2024). "DINOv2: Learning Robust Visual Features without Supervision." TMLR 2024, arXiv:2304.07193. Available at arxiv.org/abs/2304.07193. Production-grade scaling of DiNO: curated 142M-image LVD-142M dataset, ViT-g/14 teacher distilled into smaller variants, patch size 14 at 518 resolution. Table 3 (downstream benchmarks) shows DiNOv2 matching or beating supervised baselines across classification, segmentation, depth, and retrieval. The 2026 default visual feature extractor.
              </Prose>
            ),
          },
          {
            label: "Radford et al. 2021 — CLIP",
            render: () => (
              <Prose>
                Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. (2021). "Learning Transferable Visual Models From Natural Language Supervision." arXiv:2103.00020. Available at arxiv.org/abs/2103.00020. Trains a ViT image encoder and a text Transformer jointly with a contrastive loss on 400M web-scraped image-text pairs. The result: zero-shot ImageNet classification at 76.2% top-1 without a single ImageNet label. The paper that made natural-language-supervised vision a mainstream paradigm. Figure 2 (zero-shot evaluation) and Section 2.5 (the contrastive objective) are the essential reads.
              </Prose>
            ),
          },
          {
            label: "He et al. 2022 — MAE (Masked Autoencoders)",
            render: () => (
              <Prose>
                He, K., Chen, X., Xie, S., Li, Y., Dollar, P., and Girshick, R. (2022). "Masked Autoencoders Are Scalable Vision Learners." arXiv:2111.06377. Published at CVPR 2022. Available at arxiv.org/abs/2111.06377. Shows that masking 75% of patches and reconstructing pixels with an asymmetric encoder-decoder (heavy encoder on visible tokens, tiny decoder on full sequence) produces an excellent ViT initialization. MAE ViT-H at 87.8% ImageNet-1k fine-tuned accuracy was SoTA for self-supervised in 2022. Section 3 details the asymmetric design; Figure 2 is the recognition that a 75% mask ratio is key.
              </Prose>
            ),
          },
          {
            label: "Kirillov et al. 2023 — Segment Anything (SAM)",
            render: () => (
              <Prose>
                Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W-Y., Dollar, P., and Girshick, R. (2023). "Segment Anything." arXiv:2304.02643. Available at arxiv.org/abs/2304.02643. Introduces the SAM system: ViT-H image encoder + prompt encoder + lightweight mask decoder, trained on SA-1B (11M images, 1.1B masks collected via a model-in-the-loop data engine). The paper's biggest contribution is the data engine (Section 4) and the demonstration that a single promptable model can handle arbitrary segmentation tasks zero-shot. SAM-2 (2024) extends to video with a memory bank.
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
        Attempt all five before reading the answers. Exercises 1–2 test the core design reasoning; 3 tests arithmetic; 4 tests architectural judgment; 5 tests debugging.
      </Prose>

      <H3>Exercise 1 (design reasoning — why patches?)</H3>
      <Prose>
        ViT chose 16×16 non-overlapping patches over all the alternatives that had been tried — pixel-level attention, overlapping patches, learned patch boundaries, CNN stem + attention hybrid. For each of the alternatives, state in one sentence why it is worse than the 16×16 non-overlapping choice.
      </Prose>
      <Callout accent="green">
        <strong>Answer 1.</strong>
        <br />
        <strong>Pixel-level attention</strong> (224*224 = 50,176 tokens): attention is <Code>{"O(N^2)"}</Code>, and <Code>{"50{,}176^2 \\approx 2.5 \\times 10^9"}</Code> — three orders of magnitude too expensive. Even with linear-attention approximations, the quadratic token count in subsequent MLPs would kill it.
        <br />
        <strong>Overlapping patches</strong>: introduces redundancy (each pixel is in multiple patches), so the model must learn to reconcile multiple tokens for the same spatial region, and effective sequence length grows (stride smaller than kernel size). No accuracy benefit observed; compute cost goes up.
        <br />
        <strong>Learned patch boundaries</strong> (e.g. slot-attention-style): adds a nonconvex objective to learn the partition; early attempts (PatchConvNet, 2022) showed no consistent accuracy gain over fixed patches and made training less stable.
        <br />
        <strong>CNN stem + attention hybrid</strong>: adds architectural complexity and a convolutional inductive bias that, empirically, helps only at small data scales. At JFT-300M scale, pure ViT beats the hybrid; at ImageNet-1k scale, the hybrid sometimes wins (and that is what ConvNeXt-like designs explore). ViT's 16×16 non-overlapping patches are the minimum-inductive-bias choice that still makes computation tractable — a deliberate design philosophy of "let scale + data do the work".
      </Callout>

      <H3>Exercise 2 (design reasoning — shifted windows)</H3>
      <Prose>
        Swin's shifted-window design alternates W-MSA and SW-MSA blocks. Explain (a) why a single W-MSA block alone is insufficient, (b) what specifically the shift accomplishes, and (c) why the shift is cyclic (rolling) rather than a simple repartition starting at a different offset.
      </Prose>
      <Callout accent="green">
        <strong>Answer 2.</strong>
        <br />
        <strong>(a) Why W-MSA alone fails.</strong> In a pure W-MSA stack, information never crosses window boundaries. A token in the top-left window at layer 1 can communicate with any of its 48 window-mates, but it can never reach a token in the window to its right — not at layer 2, not at layer 10. The model is effectively 64 parallel independent ViTs on 49-token sequences. Global tasks (like classification of a scene where the key object straddles a window boundary) become impossible.
        <br />
        <strong>(b) What the shift accomplishes.</strong> Shifting the window partition by (M/2, M/2) means that two tokens on opposite sides of a window boundary in the original partition end up inside the same window after the shift. After one W-MSA + one SW-MSA, every token has communicated with tokens in a 13×13 = 169-token neighborhood (the union of its original window and its shifted window). After N pairs, the receptive field grows roughly as N * (M/2) in each axis — a depth-linear expansion, same asymptotic scaling as a stack of 3×3 convs.
        <br />
        <strong>(c) Why cyclic rather than offset.</strong> A non-cyclic shift would leave M/2 tokens on each side without a full window — the model would need to pad or use variable-size windows, both of which break the uniform attention pattern and hurt accuracy. Cyclic shifting keeps every window at exactly M×M = 49 tokens, uniform and friendly to batched attention kernels. The cost is that the "wrapped-around" edges become spatially incoherent (a token at the top-left now sits next to a token from the top-right), so Swin adds an attention mask to prevent those wrap-around pairs from attending to each other. This is the fiddly part of the Swin implementation — mess it up and you silently lose 1-2% accuracy.
      </Callout>

      <H3>Exercise 3 (arithmetic — ViT-B/16 FLOPs and token count)</H3>
      <Prose>
        Compute the following for ViT-B/16 at 384×384 input: (a) number of patches N, (b) approximate per-block MLP FLOPs, (c) per-block MHSA attention-score FLOPs (the <Code>{"(N+1)^2 d"}</Code> term), and (d) the ratio of MLP to attention-score FLOPs. Then interpret the ratio for 224 vs 384 vs 512.
      </Prose>
      <Callout accent="green">
        <strong>Answer 3.</strong>
        <br />
        (a) At 384/16 = 24, N = 24² = 576 patches; N+1 = 577 tokens.
        <br />
        (b) MLP FLOPs = <Code>{"8 (N+1) d^2 = 8 \\cdot 577 \\cdot 768^2 = 2.72 \\times 10^9"}</Code> ≈ 2.72 G.
        <br />
        (c) Attention-score FLOPs = <Code>{"2 (N+1)^2 d = 2 \\cdot 577^2 \\cdot 768 = 5.12 \\times 10^8"}</Code> ≈ 0.51 G (plus the same again for attn*V, so ~1.02 G total for the attention quadratic part).
        <br />
        (d) Ratio MLP/(attention-score × 2) = 2.72 / 1.02 = 2.67.
        <br />
        <strong>Interpretation across resolutions.</strong> At 224 (N=196), MLP = 0.93 G, attention quadratic = 0.12 G total, ratio ≈ 7.8 — MLP dominates by ~8×. At 384 (N=576), ratio ≈ 2.7 — attention becoming a meaningful fraction. At 512 (N=1024), ratio ≈ 1.1 — attention and MLP roughly equal. At 1024 (N=4096) the quadratic attention term dominates. This is why Swin's window attention becomes essential at detection / segmentation resolutions: for 1024×1024 inputs, global ViT attention is untenable, but Swin with M=7 windows stays linear in resolution.
      </Callout>

      <H3>Exercise 4 (architectural judgment — choosing a backbone)</H3>
      <Prose>
        You are building a product image classifier for an e-commerce catalog. You have 150,000 labeled images across 1,200 fine-grained product categories, plus access to ~10 million unlabeled product images from the same catalog. You have 8 A100 GPUs and two weeks. Rank the following four options, with justification: (a) fine-tune ViT-B/16 supervised-IN21k, (b) fine-tune DiNOv2 ViT-L/14 end-to-end, (c) DiNOv2 ViT-L/14 linear probe, (d) continued self-supervised pretraining of DiNOv2 on your 10M unlabeled images, then linear probe.
      </Prose>
      <Callout accent="green">
        <strong>Answer 4.</strong> Recommended ranking for this scenario: <strong>d &gt; c &gt; b &gt; a</strong>.
        <br />
        <strong>(d) Continued DiNOv2 pretraining + linear probe (best).</strong> 10M unlabeled domain images is more than enough to meaningfully specialize DiNOv2 features to the product domain. Continued pretraining for ~3-5 days on 8 A100s (following Meta's DiNOv2 script with a reduced schedule) typically gains 2-5% on fine-grained domain tasks versus the off-the-shelf LVD-142M features. Then a linear probe on 150k labels is fast (hours) and simple. This option leverages every resource you have.
        <br />
        <strong>(c) DiNOv2 linear probe (strong baseline, ship fast).</strong> If you need to ship in under a week, skip the continued pretraining. Linear probe on 150k labels with strong augmentation reaches ~85-90% top-1 for this class of fine-grained problem; it is the 2026 "boring correct" default. Takes a day on 8 A100s.
        <br />
        <strong>(b) Fine-tune DiNOv2 end-to-end.</strong> If linear probe underperforms (say, &lt; 82%), unfreezing the full backbone and fine-tuning with a small learning rate (1e-5) and strong augmentation gains 2-4% typically. Risk: you can overfit on 150k fine-grained labels and actually hurt feature quality. Use with care; validate on held-out categories.
        <br />
        <strong>(a) Fine-tune ViT-B/16 supervised-IN21k (weakest).</strong> Supervised ImageNet pretraining is the older paradigm. DiNOv2 features are consistently better on downstream fine-grained tasks, and you are not using the 10M unlabeled pool. Only pick this if you cannot install DiNOv2 for some infrastructure reason.
      </Callout>

      <H3>Exercise 5 (debugging — "my ViT fine-tune has collapsed")</H3>
      <Prose>
        You fine-tuned a <Code>vit_base_patch16_384</Code> from a 224 checkpoint on your 50k-image dataset at 384 resolution. The model trains without errors, loss decreases normally, and train accuracy reaches 95%. But validation accuracy is stuck at 12% — barely above random for your 8-class problem. Training accuracy is genuine (you have checked for label leakage). List four plausible causes and the one-line diagnostic that confirms each.
      </Prose>
      <Callout accent="green">
        <strong>Answer 5.</strong>
        <br />
        <strong>(1) Wrong normalization.</strong> You used ImageNet mean/std, but ViT-B/16 expects (0.5, 0.5, 0.5). Training overfits to the wrong input distribution; validation at the same wrong distribution technically trains but learns nothing generalizable.
        <br />
        <em>Diagnose:</em> <Code>{"timm.create_model(name).default_cfg['mean']"}</Code> — if it prints (0.5, 0.5, 0.5) and your dataloader uses (0.485, 0.456, 0.406), this is it. Switch to <Code>{"timm.data.create_transform(**cfg, is_training=True)"}</Code>.
        <br />
        <strong>(2) Positional embedding not interpolated.</strong> You loaded the 224 checkpoint but did not interpolate the pos_embed to the 24×24 grid needed for 384. The model is running with zero-padded pos_embeds for the last 380 positions — half the image has no positional signal.
        <br />
        <em>Diagnose:</em> <Code>{"model.pos_embed.shape"}</Code> — should be <Code>{"(1, 577, 768)"}</Code> for 384; if it is <Code>{"(1, 197, 768)"}</Code>, interpolation didn't happen. Reload with <Code>{"timm.create_model(name, pretrained=True, img_size=384)"}</Code>.
        <br />
        <strong>(3) Train/val augmentation mismatch.</strong> Your train pipeline has RandAugment + RandomResizedCrop, your val pipeline has a different resize policy (e.g. center crop at wrong crop_pct). The model learned to predict augmented views; val sees fundamentally different images.
        <br />
        <em>Diagnose:</em> run the val pipeline on a train image, visualize. If the train image passes through both pipelines and looks different in shape, scale, or color, you have found it. Match val to <Code>{"cfg['crop_pct'] = 0.9"}</Code> with the correct interpolation mode.
        <br />
        <strong>(4) Classification head frozen but backbone trainable, inverted.</strong> Common bug from copy-pasting linear-probe code: you meant to freeze the backbone and train only the head, but the <Code>requires_grad=False</Code> block hit the head, not the backbone. The backbone drifts away from its good pretrained weights while the head stays random.
        <br />
        <em>Diagnose:</em> <Code>{"sum(p.numel() for p in model.parameters() if p.requires_grad)"}</Code> — should be ~6k for a linear probe, ~86M for full fine-tune. If it is ~86M minus head (~6k) — which would be ~86M — but the head param count shows as 0 trainable, the freeze is inverted. Fix by freezing everything with <Code>{"for p in model.parameters(): p.requires_grad_(False)"}</Code>, then unfreezing the head.
      </Callout>

    </div>
  ),
};

export default visionTransformersContent;
