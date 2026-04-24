import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const depthwiseDilatedContent = {
  title: "Depthwise Separable & Dilated Convolutions",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        By late 2014, convolutional networks had solved ImageNet well enough that the question shifted from "can we make it work" to "can we make it run." AlexNet (2012) had 60M parameters. VGG-16 (2014) had 138M. GoogLeNet (2014) cut cost with inception modules but still needed a beefy server to run. Nobody was deploying these on a phone. Two architectural ideas that would eventually make real-time vision possible on a five-watt SoC were quietly invented in parallel: depthwise separable convolutions, which factor a standard conv into two cheap pieces, and dilated convolutions, which inflate a kernel's receptive field without adding parameters. Both started as theoretical constructions and became production default.
      </Prose>

      <Prose>
        The first published appearance of depthwise separable convolutions was in Laurent Sifre's 2014 PhD thesis at École Polytechnique, "Rigid-motion scattering for image classification." Sifre was studying the wavelet scattering transform, a mathematically structured alternative to learned convolutions. In a short section he observed that the expensive 2D conv over spatial-and-channel joint support could be decomposed: do the spatial filter per channel (now called <em>depthwise</em>), then mix the channels with a 1×1 conv (<em>pointwise</em>). The result was mathematically equivalent up to expressiveness constraints but dramatically cheaper in FLOPs. Sifre included experimental evidence that the factored form was competitive on CIFAR-10. The idea sat in a thesis for two years.
      </Prose>

      <Prose>
        In April 2017, Andrew Howard and colleagues at Google published "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" (arXiv 1704.04861). The paper took Sifre's factorization and built an entire ImageNet-scale classifier from it. MobileNet-1.0 achieved 70.6% top-1 ImageNet accuracy with 4.2M parameters and 569M multiply-adds — AlexNet-level accuracy (in fact slightly better) at roughly 1/14th the parameters of AlexNet and 1/20th the compute. The paper introduced two hyperparameters that are still used: a width multiplier <Code>α</Code> that scales all channel counts and a resolution multiplier <Code>ρ</Code> that scales the input image. Together they gave practitioners a single dial to trade accuracy for latency. MobileNet-1.0 became the default backbone for on-device object detection (SSD-MobileNet) and was shipped in billions of Android devices.
      </Prose>

      <Prose>
        In October 2016, François Chollet — the author of Keras — published "Xception: Deep Learning with Depthwise Separable Convolutions" (arXiv 1610.02357). Xception extended the depthwise-separable idea beyond mobile: build a server-scale ImageNet model that uses depthwise separable convolutions <em>everywhere</em> except the first layer. Chollet framed the insight theoretically: an Inception block is an intermediate point between a full conv (spatial and cross-channel correlations computed jointly) and a pair of depthwise + pointwise convs (spatial and cross-channel correlations computed independently). Xception takes this to the limit — every conv is depthwise separable — and outperforms Inception-V3 on ImageNet at similar compute. Xception is the philosophical bridge between Sifre's theoretical observation and production deployment.
      </Prose>

      <Prose>
        One year later, Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen at Google published "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (arXiv 1801.04381, CVPR 2018). MobileNetV2 introduced two structural changes that have become canonical. First, the <em>inverted residual</em>: the block <em>expands</em> channels with a 1×1 conv (typically 6×), then runs a 3×3 depthwise over the expanded tensor, then <em>projects</em> back to the low-dimensional representation with a 1×1 conv. Compare to ResNet's bottleneck, which does the opposite — reduce, compute, expand. The inversion is deliberate: depthwise is cheap per channel, so having more channels during the spatial step is nearly free, while the low-dimensional "bottleneck" states carry most of the gradient and fit better in cache. Second, <em>linear bottleneck</em>: the final 1×1 projection has no activation. Sandler argued that ReLU applied to a low-dimensional manifold destroys information; keeping the bottleneck linear preserves it. MobileNetV2 hit 72.0% top-1 at 3.4M params, and the inverted residual became the building block for nearly every efficient architecture since.
      </Prose>

      <Prose>
        In May 2019, Howard and colleagues (many overlapping with V1) published "Searching for MobileNetV3" (arXiv 1905.02244). MobileNetV3 was the first major architecture where the block structure was selected not by humans but by Neural Architecture Search (NAS) on a reward that combined accuracy and on-device latency. The paper introduced two further pieces: <em>hard-swish</em> (a piecewise-linear approximation of swish that is fast on mobile CPUs), and squeeze-and-excitation blocks inside inverted residuals. MobileNetV3-Large reached 75.2% top-1 at 5.4M params with a measured Pixel-1 latency of 51ms. This paper closed the loop: depthwise separable factorization as the primitive, NAS as the designer, latency on real hardware as the objective.
      </Prose>

      <Prose>
        The second thread of this topic is dilated convolution. In November 2015, Fisher Yu and Vladlen Koltun published "Multi-Scale Context Aggregation by Dilated Convolutions" (arXiv 1511.07122, ICLR 2016). Their motivating problem was dense prediction — semantic segmentation — where the standard image classifier architecture was a bad fit. Image classifiers pool aggressively: five stages of stride-2 or max-pool bring a 224×224 input down to 7×7 before the final classifier. This loses spatial resolution, which is fine when you only want a single label, but disastrous for per-pixel prediction. The alternatives in 2015 were either upsample back at the end (encoder-decoder, which is lossy) or remove the pooling (which kills receptive field). Yu and Koltun's fix: keep the spatial resolution but stretch the kernel. A 3×3 kernel with dilation <Code>d</Code> samples at positions <Code>{"{0, d, 2d}"}</Code> instead of <Code>{"{0, 1, 2}"}</Code> — it covers the same pattern as a 3×3 kernel zoomed out by factor <Code>d</Code>. Stacking convs at dilations 1, 2, 4, 8 gives exponential receptive field growth with linear parameter growth and unchanged spatial resolution.
      </Prose>

      <Prose>
        In June 2017, Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan Yuille at Google published "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs" (arXiv 1606.00915, TPAMI 2018). DeepLab uses dilated (they call it "atrous" — from the French "à trous", with holes) convolutions as the central spatial operation for segmentation. The paper's signature contribution is Atrous Spatial Pyramid Pooling (ASPP): at the final stage of the network, run four parallel dilated convolutions with rates 6, 12, 18, 24, plus global average pooling, then concatenate and project. This gives the classifier access to context at four spatial scales simultaneously without ever upsampling from a low-res feature map. DeepLab-V3 hit 85.7% mIoU on PASCAL VOC 2012 test — state of the art at the time, and still a baseline any segmentation paper must beat.
      </Prose>

      <Prose>
        The caveat came in February 2018, when Panqu Wang and colleagues at TuSimple published "Understanding Convolution for Semantic Segmentation" (arXiv 1702.08502). The paper pointed out a failure mode of naive dilated stacks called the <em>gridding artifact</em>: when you stack several dilated convs all at the same rate, the receptive field samples a sparse grid rather than a dense region, because successive kernels land on exactly the same subgrid. The paper proposed Hybrid Dilated Convolution (HDC): use co-prime dilation rates like (1, 2, 3) or (1, 2, 5) so consecutive layers cover each other's gaps. HDC + hybrid classification schemes lifted Cityscapes mIoU by 1–2 points over naive dilation. Every serious segmentation architecture since DeepLab V3+ uses some variation of HDC-style rate selection.
      </Prose>

      <Prose>
        By 2026 the picture is: depthwise separable convolutions are the default for mobile and edge deployment (MobileNet, EfficientNet, RegNet, ConvNeXt variants), while dilated convolutions remain the default spatial operator for dense prediction (DeepLab, segmentation transformers' patch-mixing stages). Both primitives are now being re-examined through the lens of vision transformers — but even there, the convolutional stem of most production ViTs still uses depthwise separable blocks, and segmentation heads still use ASPP. The factorizations these two lines of work introduced are foundational.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Both ideas are factorizations of a standard convolution. Understanding them means seeing clearly what a standard conv is doing and which part is expensive.
      </Prose>

      <Prose>
        A standard 2D convolution with kernel size <Code>k</Code>, <Code>{"C_{in}"}</Code> input channels, and <Code>{"C_{out}"}</Code> output channels computes each output channel as a sum over all input channels of a spatial convolution:
      </Prose>

      <MathBlock>
        {"y_{c_{out}}[i, j] = \\sum_{c_{in}=1}^{C_{in}} \\sum_{u, v} x_{c_{in}}[i+u, j+v] \\cdot w[c_{out}, c_{in}, u, v]"}
      </MathBlock>

      <Prose>
        Two things happen at once in that double sum. The inner sum over <Code>(u, v)</Code> is the <em>spatial</em> operation — finding a pattern in a local neighborhood. The outer sum over <Code>{"c_{in}"}</Code> is the <em>cross-channel</em> operation — mixing information across feature maps. A standard conv does both jointly; every weight lives in a 4D block of shape <Code>{"[C_{out}, C_{in}, k, k]"}</Code>, and every output channel sees every input channel's spatial response.
      </Prose>

      <Prose>
        <strong>Depthwise separable idea — factor the two axes.</strong> Depthwise separable convolution performs the spatial operation and the cross-channel operation in sequence, not together. The depthwise stage applies one <Code>k × k</Code> filter per input channel; the pointwise stage applies a 1×1 convolution to mix channels. Together:
      </Prose>

      <MathBlock>
        {"y_{c_{out}}[i, j] = \\sum_{c=1}^{C_{in}} w_{pw}[c_{out}, c] \\cdot \\left( \\sum_{u, v} x_c[i+u, j+v] \\cdot w_{dw}[c, u, v] \\right)"}
      </MathBlock>

      <Prose>
        The outer weight <Code>{"w_{pw}"}</Code> is a 2D matrix of shape <Code>{"[C_{out}, C_{in}]"}</Code>; the inner weight <Code>{"w_{dw}"}</Code> is a 3D tensor of shape <Code>{"[C_{in}, k, k]"}</Code>. Standard conv uses <Code>{"C_{out} \\cdot C_{in} \\cdot k^2"}</Code> weights; depthwise separable uses <Code>{"C_{in} \\cdot k^2 + C_{out} \\cdot C_{in}"}</Code>. For typical settings (<Code>k = 3</Code>, <Code>{"C_{out} = 64"}</Code>) this is an 8–9× reduction in both parameters and FLOPs. The cost is a modest loss of expressive power: the joint 4D tensor of a standard conv cannot always be written exactly as the product of a depthwise kernel and a pointwise matrix (it would need to be low-rank in a specific sense). In practice, the gap is small enough that depthwise separable beats the larger standard conv per unit of compute budget.
      </Prose>

      <Prose>
        <strong>Dilated idea — enlarge the kernel without filling it.</strong> A standard <Code>k × k</Code> kernel samples a contiguous block. A dilated kernel samples a spread-out version of the same block, inserting <Code>d - 1</Code> "holes" (skipped positions) between taps. Formally:
      </Prose>

      <MathBlock>
        {"y[i, j] = \\sum_{u, v} x[i + d \\cdot u, j + d \\cdot v] \\cdot w[u, v]"}
      </MathBlock>

      <Prose>
        The effective kernel size — the spatial extent the conv reaches into — is <Code>{"(k - 1) \\cdot d + 1"}</Code>. At <Code>d = 2</Code> a 3×3 kernel covers a 5×5 region; at <Code>d = 4</Code> it covers 9×9; at <Code>d = 8</Code>, 17×17. The parameter count is unchanged: still <Code>{"k^2"}</Code> weights. What increases is the <em>receptive field</em>, which is what matters for dense prediction tasks where each output pixel needs to aggregate information from a large surrounding context (is this pixel sky or road? depends on the whole lower third of the image).
      </Prose>

      <Prose>
        <strong>Inverted residual idea — dual of the ResNet bottleneck.</strong> MobileNetV2's block flips the ResNet bottleneck inside out. A ResNet bottleneck has the shape "expand → compute → contract" only if you consider the 1×1 convs as expansion/contraction around the 3×3; in fact ResNet does <em>contract → compute → expand</em>: 1×1 reduces channels by 4× to a bottleneck representation, 3×3 operates there, 1×1 expands back. The input and output of the block are the "wide" representation (e.g., 256 channels), and the internal compute runs at the "narrow" representation (64). MobileNetV2 inverts this: input and output are <em>narrow</em> (e.g., 32 channels), and the block internally expands to <em>wide</em> (6× = 192 channels) via a 1×1 conv, does depthwise 3×3 on the wide tensor, then projects back to narrow. The depthwise stage is cheap per channel, so having 192 channels in the middle is affordable; the narrow inputs/outputs mean the skip connection and the 1×1 projections are themselves cheap.
      </Prose>

      <Callout accent="gold">
        Mental model: depthwise separable factors a conv spatially-then-channel-wise; dilation factors a conv at two scales of spatial support without paying the parameter cost. Inverted residuals apply the first factorization inside a skip-connected block organized around a narrow pass-through representation. These three ideas compose: EfficientNet, MobileNetV3, RegNet-Y, and the ConvNeXt family all sit at intersections of "depthwise separable + inverted residual + modern regularization."
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Cost of a standard convolution</H3>

      <Prose>
        A standard <Code>k × k</Code> convolution mapping <Code>{"C_{in}"}</Code> input channels to <Code>{"C_{out}"}</Code> output channels, applied to a feature map of spatial size <Code>H × W</Code>, uses:
      </Prose>

      <MathBlock>
        {"\\text{params}_{\\text{std}} = k^2 \\cdot C_{in} \\cdot C_{out}"}
      </MathBlock>

      <MathBlock>
        {"\\text{FLOPs}_{\\text{std}} = k^2 \\cdot C_{in} \\cdot C_{out} \\cdot H \\cdot W"}
      </MathBlock>

      <Prose>
        Both parameters and FLOPs scale as the product of all four dimensions — kernel size squared, input channels, output channels, and spatial extent. This quadratic dependence on channels is what makes the standard conv expensive at high channel counts (think 512 → 1024 in a deep ResNet stage).
      </Prose>

      <H3>3.2 Cost of a depthwise separable convolution</H3>

      <Prose>
        Depthwise separable splits the operation into two stages. The depthwise stage applies <Code>{"C_{in}"}</Code> filters of size <Code>k × k</Code>, one per input channel (no cross-channel interaction). The pointwise stage applies a 1×1 convolution from <Code>{"C_{in}"}</Code> to <Code>{"C_{out}"}</Code> channels. Summing:
      </Prose>

      <MathBlock>
        {"\\text{params}_{\\text{sep}} = k^2 \\cdot C_{in} + C_{in} \\cdot C_{out}"}
      </MathBlock>

      <MathBlock>
        {"\\text{FLOPs}_{\\text{sep}} = k^2 \\cdot C_{in} \\cdot H \\cdot W + C_{in} \\cdot C_{out} \\cdot H \\cdot W"}
      </MathBlock>

      <Prose>
        The ratio of separable to standard cost is:
      </Prose>

      <MathBlock>
        {"\\frac{\\text{FLOPs}_{\\text{sep}}}{\\text{FLOPs}_{\\text{std}}} = \\frac{1}{C_{out}} + \\frac{1}{k^2}"}
      </MathBlock>

      <Prose>
        For <Code>k = 3</Code> the term <Code>{"1 / k^2 = 1/9 \\approx 0.111"}</Code> dominates whenever <Code>{"C_{out}"}</Code> is not tiny. For <Code>{"C_{out} = 64"}</Code> the ratio is <Code>{"1/64 + 1/9 \\approx 0.127"}</Code> — a 7.9× reduction. For <Code>{"C_{out} = 256"}</Code> it drops to 0.115, giving 8.7×. Asymptotically (<Code>{"C_{out} \\to \\infty"}</Code>) the ratio approaches <Code>{"1/k^2"}</Code>, so a 3×3 depthwise separable is 9× cheaper than a 3×3 standard conv in the limit. The FLOP reduction is a fixed factor set by the kernel size, independent of channel count — a remarkable property.
      </Prose>

      <H3>3.3 Dilated convolution formulation</H3>

      <Prose>
        A dilated convolution with dilation rate <Code>d</Code> replaces the standard sampling pattern with a stretched one:
      </Prose>

      <MathBlock>
        {"y[i, j] = \\sum_{u=0}^{k-1} \\sum_{v=0}^{k-1} x[i + d \\cdot u, j + d \\cdot v] \\cdot w[u, v]"}
      </MathBlock>

      <Prose>
        The effective spatial extent of the kernel — how far apart the leftmost and rightmost taps are — is:
      </Prose>

      <MathBlock>
        {"k_{\\text{eff}} = (k - 1) \\cdot d + 1"}
      </MathBlock>

      <Prose>
        For a 3×3 kernel at dilation 1, 2, 4, 8 the effective extents are 3, 5, 9, 17. The number of weights in the kernel is still <Code>{"k^2 = 9"}</Code> for all of them. Dilation adds no parameters and only trivial FLOPs (the index arithmetic to skip into the input tensor).
      </Prose>

      <H3>3.4 Receptive field growth with dilation</H3>

      <Prose>
        The receptive field of a layer is the set of input pixels that can influence a given output pixel. For a stack of <Code>L</Code> convolutions at dilations <Code>{"d_1, d_2, \\ldots, d_L"}</Code> and kernel size <Code>k</Code>, the receptive field size grows as:
      </Prose>

      <MathBlock>
        {"\\text{RF}_L = 1 + \\sum_{\\ell=1}^{L} (k - 1) \\cdot d_\\ell"}
      </MathBlock>

      <Prose>
        With all dilations equal to 1 (standard convs), the RF grows linearly: a stack of 10 convs of kernel 3 gives <Code>{"\\text{RF} = 1 + 10 \\cdot 2 = 21"}</Code> pixels. With exponential dilations <Code>{"d_\\ell = 2^{\\ell-1}"}</Code>, the RF grows geometrically:
      </Prose>

      <MathBlock>
        {"\\text{RF}_L = 1 + (k - 1) \\cdot (1 + 2 + 4 + \\ldots + 2^{L-1}) = 1 + (k - 1) \\cdot (2^L - 1)"}
      </MathBlock>

      <Prose>
        For <Code>k = 3</Code>, <Code>L = 7</Code>: <Code>{"\\text{RF} = 1 + 2 \\cdot 127 = 255"}</Code> pixels. A seven-layer dilated stack covers a 255×255 window; a seven-layer plain conv stack covers only 15×15. This is the engine behind dilated segmentation networks: get a huge receptive field without stride or pooling.
      </Prose>

      <H3>3.5 Gridding: the failure mode of stacked equal dilations</H3>

      <Prose>
        If every layer in a dilated stack uses the same dilation <Code>d</Code>, the taps of the effective sampling pattern land on a sparse sublattice rather than densely covering the receptive field. Concretely, a stack of three <Code>k = 3</Code> convs all at <Code>d = 2</Code> produces an effective sampling pattern of size 13×13 that only covers 49 of those 169 pixels (29%) — the rest are never read. This is the <em>gridding artifact</em> (Wang et al. 2018). The network sees a checkerboard-sparse input; features of small objects smaller than the stride can be invisible.
      </Prose>

      <Prose>
        The fix is Hybrid Dilated Convolution (HDC): use a sequence of dilation rates whose greatest common divisor is 1. Rates like <Code>{"(1, 2, 3)"}</Code>, <Code>{"(1, 2, 5)"}</Code>, or <Code>{"(3, 4, 5)"}</Code> produce full dense coverage of the receptive field. The paper proves that a sufficient condition for full coverage is:
      </Prose>

      <MathBlock>
        {"M_\\ell = \\max\\left[M_{\\ell+1} - 2 r_\\ell, \\; M_{\\ell+1} - 2 (M_{\\ell+1} - r_\\ell), \\; r_\\ell\\right] \\leq k \\text{ for all } \\ell"}
      </MathBlock>

      <Prose>
        where <Code>{"M_\\ell"}</Code> is the "maximum distance between two non-zero values" at layer <Code>{"\\ell"}</Code> and <Code>{"r_\\ell"}</Code> is the dilation at that layer. In plain English: adjacent layers' dilations should not be multiples of one another.
      </Prose>

      <H3>3.6 Inverted residual block</H3>

      <Prose>
        The MobileNetV2 block with expansion ratio <Code>t</Code> computes:
      </Prose>

      <MathBlock>
        {"\\mathbf{h}_1 = \\text{ReLU6}(\\text{BN}(W_{\\text{expand}} \\ast \\mathbf{x})), \\quad \\mathbf{h}_1 \\in \\mathbb{R}^{t C \\times H \\times W}"}
      </MathBlock>

      <MathBlock>
        {"\\mathbf{h}_2 = \\text{ReLU6}(\\text{BN}(W_{\\text{dw}} \\ast_{\\text{group}=tC} \\mathbf{h}_1))"}
      </MathBlock>

      <MathBlock>
        {"\\mathbf{y} = \\text{BN}(W_{\\text{project}} \\ast \\mathbf{h}_2) + \\mathbf{x}"}
      </MathBlock>

      <Prose>
        The expansion 1×1 conv goes from <Code>C</Code> to <Code>tC</Code> channels (typically <Code>t = 6</Code>). The depthwise 3×3 acts on the expanded tensor with <Code>groups = tC</Code>. The final 1×1 projects back to <Code>C</Code> with <em>no activation</em> (linear bottleneck). The residual connection adds the original narrow input back to the projected output. The design is counter-intuitive relative to ResNet but perfectly tuned for depthwise separable costs: the expensive parts scale with <Code>tC</Code>, but <Code>t</Code> cancels out in the product thanks to depthwise being O(channels), not O(channels²).
      </Prose>

      <H3>3.7 Atrous Spatial Pyramid Pooling (ASPP)</H3>

      <Prose>
        DeepLab V2/V3's ASPP module processes a feature map with four parallel dilated convs at rates <Code>{"r \\in \\{6, 12, 18\\}"}</Code> plus global average pooling, concatenates them, and reduces back with a 1×1:
      </Prose>

      <MathBlock>
        {"\\text{ASPP}(\\mathbf{x}) = W_{\\text{out}} \\ast \\text{Concat}\\left[\\text{DConv}_{r=1}(\\mathbf{x}), \\; \\text{DConv}_{r=6}(\\mathbf{x}), \\; \\text{DConv}_{r=12}(\\mathbf{x}), \\; \\text{DConv}_{r=18}(\\mathbf{x}), \\; \\text{GAP}(\\mathbf{x})\\right]"}
      </MathBlock>

      <Prose>
        Each branch sees the same spatial resolution but a different receptive-field scale. The concatenation gives the classifier multi-scale context without pyramid pooling's downsampling. ASPP is what makes DeepLab output pixel-level predictions at 1/8 or 1/16 of input resolution while still seeing context from the whole image.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All four snippets below were run locally on PyTorch 2.6. Outputs are verbatim stdout.
      </Prose>

      <H3>4a. Depthwise + pointwise from scratch — verify identity to nn.Conv2d</H3>

      <Prose>
        A depthwise separable conv is a grouped 3×3 (groups = <Code>{"C_{in}"}</Code>) followed by a 1×1. The test below builds it from raw tensors, compares against <Code>nn.Conv2d(..., groups=Cin)</Code>, and verifies the output is bit-exact. Then we compute the parameter and FLOP counts vs the standard conv to confirm the <Code>{"1/C_{out} + 1/k^2"}</Code> formula.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

def dw_sep_conv(x, dw_weight, pw_weight):
    # x          : [N, Cin, H, W]
    # dw_weight  : [Cin, 1, k, k]     one kxk filter per input channel
    # pw_weight  : [Cout, Cin, 1, 1]  1x1 channel mixer
    N, Cin, H, W = x.shape
    k = dw_weight.shape[-1]
    # Depthwise: groups=Cin means each input channel gets its own filter
    dw = F.conv2d(x, dw_weight, stride=1, padding=k // 2, groups=Cin)
    # Pointwise: standard 1x1 conv, no padding needed
    out = F.conv2d(dw, pw_weight, stride=1, padding=0)
    return out

N, Cin, Cout, H, W, k = 1, 4, 8, 5, 5, 3
x = torch.randn(N, Cin, H, W)
dw_w = torch.randn(Cin, 1, k, k)
pw_w = torch.randn(Cout, Cin, 1, 1)

out_scratch = dw_sep_conv(x, dw_w, pw_w)

dw_layer = nn.Conv2d(Cin, Cin, k, padding=k // 2, groups=Cin, bias=False)
pw_layer = nn.Conv2d(Cin, Cout, 1, bias=False)
with torch.no_grad():
    dw_layer.weight.copy_(dw_w)
    pw_layer.weight.copy_(pw_w)
out_layer = pw_layer(dw_layer(x))

std_params = k * k * Cin * Cout
sep_params = k * k * Cin + Cin * Cout
std_flops  = k * k * Cin * Cout * H * W
sep_flops  = k * k * Cin * H * W + Cin * Cout * H * W

print("Depthwise separable vs standard conv")
print("=" * 50)
print(f"shapes: x={tuple(x.shape)}  out={tuple(out_scratch.shape)}")
print(f"max abs diff (scratch vs nn.Conv2d) : {(out_scratch - out_layer).abs().max().item():.2e}")
print()
print(f"params  : standard={std_params:5d}   separable={sep_params:5d}   ratio={std_params/sep_params:.2f}x")
print(f"FLOPs   : standard={std_flops:5d}   separable={sep_flops:5d}   ratio={std_flops/sep_flops:.2f}x")
print(f"theory  : 1/Cout + 1/k^2 = {1/Cout + 1/(k*k):.4f} -> {std_flops/sep_flops:.2f}x reduction")

# Output:
# Depthwise separable vs standard conv
# ==================================================
# shapes: x=(1, 4, 5, 5)  out=(1, 8, 5, 5)
# max abs diff (scratch vs nn.Conv2d) : 0.00e+00
#
# params  : standard=  288   separable=   68   ratio=4.24x
# FLOPs   : standard= 7200   separable= 1700   ratio=4.24x
# theory  : 1/Cout + 1/k^2 = 0.2361 -> 4.24x reduction`}
      </CodeBlock>

      <Prose>
        The scratch implementation matches <Code>nn.Conv2d</Code>'s grouped conv bit-exactly. The ratio 4.24× matches the closed-form prediction <Code>{"1 / 0.2361"}</Code>. Note: for small <Code>{"C_{out}"}</Code> like 8, the <Code>{"1 / C_{out}"}</Code> term is not yet negligible; the asymptotic 9× limit is only reached when <Code>{"C_{out} \\gtrsim 64"}</Code>. The next snippet shows the ratio at realistic MobileNet-scale channel counts.
      </Prose>

      <H3>4b. FLOP ratio at realistic channel counts</H3>

      <CodeBlock language="python">
{`rows = []
for Cin, Cout in [(32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]:
    k = 3
    std = k * k * Cin * Cout
    sep = k * k * Cin + Cin * Cout
    rows.append((Cin, Cout, std, sep, std / sep))

print(f"{'Cin':>5}  {'Cout':>5}  {'standard':>10}  {'separable':>10}  {'ratio':>6}")
for Cin, Cout, std, sep, r in rows:
    print(f"{Cin:>5}  {Cout:>5}  {std:>10}  {sep:>10}  {r:>5.2f}x")

# Output:
#   Cin   Cout    standard   separable   ratio
#    32     64       18432        2336   7.89x
#    64    128       73728        8768   8.41x
#   128    256      294912       33920   8.69x
#   256    512     1179648      133376   8.84x
#   512   1024     4718592      528896   8.92x`}
      </CodeBlock>

      <Prose>
        At production channel counts (512 → 1024) the ratio is 8.92× — very close to the asymptotic 9× for <Code>k = 3</Code>. This is the canonical MobileNet claim: "8–9× fewer FLOPs than a standard conv of the same input/output shape."
      </Prose>

      <H3>4c. Dilated conv from scratch — verify and plot receptive field</H3>

      <Prose>
        The scratch dilated conv below uses explicit index arithmetic <Code>{"x[i + d \\cdot u, j + d \\cdot v]"}</Code> and compares against <Code>nn.Conv2d(..., dilation=d)</Code>. Then we compute the RF growth of a stack at dilations 1, 2, 4, 8, 16, 32, 64.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)

def dilated_conv_scratch(x, w, d=1):
    # x : [N, Cin, H, W]    w : [Cout, Cin, k, k]
    N, Cin, H, W = x.shape
    Cout, _, k, _ = w.shape
    eff = (k - 1) * d + 1
    pad = eff // 2
    xp = F.pad(x, (pad, pad, pad, pad))
    out = torch.zeros(N, Cout, H, W)
    for i in range(H):
        for j in range(W):
            for u in range(k):
                for v in range(k):
                    xi = i + u * d
                    xj = j + v * d
                    patch = xp[:, :, xi, xj]             # [N, Cin]
                    out[:, :, i, j] += patch @ w[:, :, u, v].T
    return out

N, Cin, Cout, H, W, k = 1, 2, 3, 7, 7, 3
x = torch.randn(N, Cin, H, W)
w = torch.randn(Cout, Cin, k, k)

for d in [1, 2, 3]:
    out_scratch = dilated_conv_scratch(x, w, d)
    layer = nn.Conv2d(Cin, Cout, k, dilation=d, padding=((k - 1) * d + 1) // 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(w)
    out_ref = layer(x)
    eff = (k - 1) * d + 1
    print(f"dilation={d}  eff_kernel={eff}x{eff}  max_abs_err={(out_scratch - out_ref).abs().max().item():.2e}")

print()
print("Receptive field of a stack of k=3 convs with exponential dilation 1,2,4,...")
rf = 1
k = 3
for layer_idx in range(7):
    d = 2 ** layer_idx
    rf = rf + (k - 1) * d
    print(f"  layer {layer_idx+1}: dilation={d:3d}  RF={rf}")

# Output:
# dilation=1  eff_kernel=3x3  max_abs_err=9.54e-07
# dilation=2  eff_kernel=5x5  max_abs_err=9.54e-07
# dilation=3  eff_kernel=7x7  max_abs_err=9.54e-07
#
# Receptive field of a stack of k=3 convs with exponential dilation 1,2,4,...
#   layer 1: dilation=  1  RF=3
#   layer 2: dilation=  2  RF=7
#   layer 3: dilation=  4  RF=15
#   layer 4: dilation=  8  RF=31
#   layer 5: dilation= 16  RF=63
#   layer 6: dilation= 32  RF=127
#   layer 7: dilation= 64  RF=255`}
      </CodeBlock>

      <Prose>
        The scratch dilated conv agrees with PyTorch's <Code>dilation=d</Code> argument to float precision (1e-6 is standard float32 accumulation error). The RF doubles every layer under exponential dilation — 7 layers cover 255 pixels, which is larger than the entire 224×224 ImageNet input. A network with this spatial coverage can classify "what's in this image" by the last layer even though every layer preserves the full resolution.
      </Prose>

      <H3>4d. Gridding artifact — dilations (2, 2, 2) vs HDC (1, 2, 3)</H3>

      <Prose>
        This snippet forwards coverage masks through a stack of three <Code>k=3</Code> dilated convs and shows which input pixels are actually sampled. A <Code>#</Code> means the output pixel at the center depends on that input; <Code>.</Code> means no contribution path exists. Equal dilations produce a checkerboard; co-prime rates fill the window densely.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn.functional as F

def receptive_field_coverage(dilations, k=3):
    size = 1
    mask = torch.ones(1, 1, 1, 1)
    for d in dilations:
        w = torch.ones(1, 1, k, k)       # unit kernel: every tap contributes
        mask = F.conv_transpose2d(mask, w, stride=1, padding=0, dilation=d)
        mask = (mask > 0).float()
    return mask.squeeze().int()

def show(mask, title):
    print(title)
    H, W = mask.shape
    for row in mask.tolist():
        print("  " + "".join("#" if v else "." for v in row))
    covered = int(mask.sum().item())
    total = H * W
    print(f"  coverage: {covered}/{total} = {100*covered/total:.1f}%")
    print()

show(receptive_field_coverage([2, 2, 2]),
     "Dilations [2, 2, 2] - naive stacked dilation (GRIDDING):")
show(receptive_field_coverage([1, 2, 3]),
     "Dilations [1, 2, 3] - HDC (Wang 2018, all pixels covered):")

# Output:
# Dilations [2, 2, 2] - naive stacked dilation (GRIDDING):
#   #.#.#.#.#.#.#
#   .............
#   #.#.#.#.#.#.#
#   .............
#   #.#.#.#.#.#.#
#   .............
#   #.#.#.#.#.#.#
#   .............
#   #.#.#.#.#.#.#
#   .............
#   #.#.#.#.#.#.#
#   .............
#   #.#.#.#.#.#.#
#   coverage: 49/169 = 29.0%
#
# Dilations [1, 2, 3] - HDC (Wang 2018, all pixels covered):
#   #############
#   #############
#   #############
#   #############
#   #############
#   #############
#   #############
#   #############
#   #############
#   #############
#   #############
#   #############
#   #############
#   coverage: 169/169 = 100.0%`}
      </CodeBlock>

      <Prose>
        The [2, 2, 2] stack samples 29% of its receptive field — three-quarters of the pixels inside the 13×13 window are never read. This is what Wang et al. 2018 called gridding: the network can literally be blind to small features whose support falls in the off-grid positions. HDC at [1, 2, 3] covers 100% of the same window with the same parameter count. The lesson has propagated through every modern segmentation architecture; you will not find a production DeepLab-family model with equal stacked dilations.
      </Prose>

      <H3>4e. MobileNetV2 inverted residual from scratch and FLOP sweep</H3>

      <Prose>
        The block below is the canonical MobileNetV2 InvertedResidual: 1×1 expand (t=6) → 3×3 depthwise → 1×1 project (linear) + skip. We compare against a standard 3×3 conv block and a MobileNetV1 depthwise-separable block at the same channel count. Compute is measured on CPU (where mobile models actually run).
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn, torch.nn.functional as F, time
torch.manual_seed(0)

class StandardBlock(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.conv = nn.Conv2d(C, C, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(C)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class DWSepBlock(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.dw  = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)
        self.bn1 = nn.BatchNorm2d(C)
        self.pw  = nn.Conv2d(C, C, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(C)
    def forward(self, x):
        h = F.relu(self.bn1(self.dw(x)))
        return F.relu(self.bn2(self.pw(h)))

class InvertedResidual(nn.Module):
    def __init__(self, C, expand=6):
        super().__init__()
        Cm = C * expand
        self.expand = nn.Conv2d(C, Cm, 1, bias=False); self.bn1 = nn.BatchNorm2d(Cm)
        self.dw     = nn.Conv2d(Cm, Cm, 3, padding=1, groups=Cm, bias=False); self.bn2 = nn.BatchNorm2d(Cm)
        self.project = nn.Conv2d(Cm, C, 1, bias=False); self.bn3 = nn.BatchNorm2d(C)
    def forward(self, x):
        h = F.relu6(self.bn1(self.expand(x)))
        h = F.relu6(self.bn2(self.dw(h)))
        h = self.bn3(self.project(h))       # linear bottleneck
        return h + x                        # residual

def count_params(m): return sum(p.numel() for p in m.parameters())
def conv_flops(m, H, W):
    total = 0
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            Cin, Cout, k, g = mod.in_channels, mod.out_channels, mod.kernel_size[0], mod.groups
            total += (k * k * Cin * Cout // g) * H * W
    return total

C, H, W = 96, 14, 14
blocks = {
    "Standard 3x3 conv"       : StandardBlock(C),
    "MBv1 depthwise separable": DWSepBlock(C),
    "MBv2 inverted residual"  : InvertedResidual(C),
}

x = torch.randn(1, C, H, W)
def bench(m, n=500):
    m.eval()
    with torch.no_grad():
        for _ in range(10): _ = m(x)
        t0 = time.perf_counter()
        for _ in range(n): _ = m(x)
        return (time.perf_counter() - t0) / n * 1e6

print(f"input: [1, {C}, {H}, {W}]")
print(f"{'block':<26}  {'params':>8}  {'FLOPs':>10}  {'CPU us':>8}")
for name, m in blocks.items():
    print(f"{name:<26}  {count_params(m):>8}  {conv_flops(m, H, W):>10}  {bench(m):>8.1f}")

# Output:
# input: [1, 96, 14, 14]
# block                         params       FLOPs    CPU us
# Standard 3x3 conv              83136    16257024     365.5
# MBv1 depthwise separable       10464     1975680     427.7
# MBv2 inverted residual        118272    22692096    1094.5`}
      </CodeBlock>

      <Prose>
        The depthwise-separable block (MBv1 style) has 8× fewer parameters and 8.2× fewer FLOPs than the standard 3×3 block. Yet CPU latency is slightly <em>worse</em> (428 vs 366 microseconds) because the kernel is split into two ops and the depthwise op is memory-bandwidth-bound on desktop CPUs — a theme that recurs in section 8. The MobileNetV2 inverted residual has <em>more</em> params and FLOPs than the plain block because of the 6× expansion, yet this is the block that wins at mobile deployment because expansion/depthwise/projection fuses cleanly in TFLite and NNAPI. FLOP count is a misleading proxy for on-device latency.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5.1 Depthwise and pointwise layers in torch.nn</H3>

      <Prose>
        PyTorch's <Code>nn.Conv2d</Code> handles both operations via standard arguments. Depthwise is a grouped conv with <Code>{"groups = C_{in}"}</Code>; pointwise is a 1×1 conv:
      </Prose>

      <CodeBlock language="python">
{`import torch.nn as nn

# Depthwise 3x3 conv: one filter per input channel, no channel mixing
dw = nn.Conv2d(
    in_channels=96, out_channels=96,
    kernel_size=3, padding=1,
    groups=96,       # groups == in_channels == out_channels  -> depthwise
    bias=False,
)

# Pointwise 1x1 conv: channel mixing, no spatial support
pw = nn.Conv2d(
    in_channels=96, out_channels=128,
    kernel_size=1,
    bias=False,
)

# Full depthwise separable block, typical MobileNetV1 pattern
class DWSep(nn.Module):
    def __init__(self, Cin, Cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(Cin, Cin, 3, stride=stride, padding=1,
                             groups=Cin, bias=False)
        self.bn1 = nn.BatchNorm2d(Cin)
        self.pw = nn.Conv2d(Cin, Cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(Cout)
    def forward(self, x):
        x = nn.functional.relu6(self.bn1(self.dw(x)))
        x = nn.functional.relu6(self.bn2(self.pw(x)))
        return x`}
      </CodeBlock>

      <H3>5.2 Dilated conv via the dilation argument</H3>

      <CodeBlock language="python">
{`import torch.nn as nn

# 3x3 dilated conv with rate 2. Effective kernel extent is (3-1)*2 + 1 = 5.
# Padding must match: pad = ((k-1) * d) // 2 to keep spatial size.
dconv = nn.Conv2d(
    in_channels=256, out_channels=256,
    kernel_size=3,
    dilation=2,
    padding=2,       # (k-1) * d / 2 = 2
    bias=False,
)

# Dilated depthwise (used in EfficientNet-Edge and some DeepLab variants)
d_dw = nn.Conv2d(
    in_channels=256, out_channels=256,
    kernel_size=3,
    dilation=4, padding=4,
    groups=256,
    bias=False,
)`}
      </CodeBlock>

      <H3>5.3 MobileNetV2 InvertedResidual from torchvision</H3>

      <Prose>
        The reference implementation lives in <Code>torchvision.models.mobilenetv2</Code>. The block is parameterized by input channels, output channels, stride, and expansion ratio:
      </Prose>

      <CodeBlock language="python">
{`# Simplified from torchvision/models/mobilenetv2.py

import torch.nn as nn

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden = int(round(inp * expand_ratio))
        self.use_res = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:                          # 1x1 expand
            layers += [
                nn.Conv2d(inp, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]
        layers += [                                    # 3x3 depthwise
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1,
                      groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, oup, 1, bias=False),     # 1x1 project (linear)
            nn.BatchNorm2d(oup),                       # no activation
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        return x + out if self.use_res else out`}
      </CodeBlock>

      <Prose>
        Three design details are worth noting. First, the residual is only added when stride is 1 AND input/output channels match — downsampling or channel-change blocks run without a skip. Second, the final BatchNorm on the projection has no ReLU — this is Sandler et al.'s linear bottleneck, empirically critical for preserving information across narrow representations. Third, ReLU6 (min(max(x, 0), 6)) is used instead of ReLU; the clipping at 6 helps fixed-point quantization, which is non-negotiable for on-device deployment.
      </Prose>

      <H3>5.4 DeepLab ASPP module</H3>

      <Prose>
        DeepLab V3's ASPP concatenates 1×1 conv, three dilated 3×3 convs at rates (6, 12, 18), and a global average pooling branch. The implementation below mirrors <Code>torchvision.models.segmentation.deeplabv3.ASPP</Code>:
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn, torch.nn.functional as F

class ASPPConv(nn.Sequential):
    def __init__(self, in_c, out_c, dilation):
        super().__init__(
            nn.Conv2d(in_c, out_c, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

class ASPPPooling(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        size = x.shape[-2:]
        h = F.relu(self.bn(self.conv(self.gap(x))))
        return F.interpolate(h, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_c, out_c=256, rates=(6, 12, 18)):
        super().__init__()
        modules = [
            nn.Sequential(nn.Conv2d(in_c, out_c, 1, bias=False),
                          nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        ]
        modules += [ASPPConv(in_c, out_c, r) for r in rates]
        modules.append(ASPPPooling(in_c, out_c))
        self.branches = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        return self.project(torch.cat(feats, dim=1))`}
      </CodeBlock>

      <Prose>
        Three parallel dilated branches + one 1×1 + one global-pool branch = five branches, concatenated along channels, projected back to <Code>out_c</Code>. Rates 6/12/18 were chosen by Chen et al. for output stride 16 (the spatial downsampling factor between input and ASPP's input). For output stride 8, the paper doubles them to 12/24/36. The rates scale with output stride so that the equivalent receptive field in input-pixel units is fixed.
      </Prose>

      <H3>5.5 timm's EfficientNet blocks</H3>

      <Prose>
        In Ross Wightman's <Code>timm</Code> library, the canonical "MBConv" block is parameterized and reused across MobileNetV2/V3, EfficientNet, EfficientNet-V2, MobileViT, and ConvNeXt-derivatives. A typical call:
      </Prose>

      <CodeBlock language="python">
{`import timm

# EfficientNet-B0 has 16 MBConv blocks with various expansion ratios and kernel sizes
m = timm.create_model('efficientnet_b0', pretrained=True)

# Inspect a typical block (stage 2, block 0)
block = m.blocks[2][0]
print(type(block).__name__)    # InvertedResidual
# Structure:
#   conv_pw (1x1 expand)    -> bn1 -> act1 (swish)
#   conv_dw (3x3 depthwise) -> bn2 -> act2 (swish)
#   se (squeeze-excitation, MobileNetV3+ feature)
#   conv_pwl (1x1 linear project) -> bn3
#   residual if stride==1 and in==out

# Training this from scratch on ImageNet at 224x224 needs ~ 350 GPU-hours on A100.
# Pretrained weights are always downloaded in practice.`}
      </CodeBlock>

      <Callout accent="gold">
        Production rule-of-thumb for 2026: use <Code>timm</Code> for any MobileNet / EfficientNet / ConvNeXt variant rather than hand-rolling blocks. The library handles the 10+ subtle details (stochastic depth, drop-path, squeeze-excitation, h-swish, fused-MBConv for early stages) that hand-rolled code will get wrong. For dense prediction, use <Code>torchvision.models.segmentation</Code> for DeepLab or SegFormer from HuggingFace. For edge deployment, convert to ONNX or TFLite and quantize with per-channel int8 — depthwise-separable nets quantize especially well.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Depthwise vs standard conv kernel structure</H3>

      <Prose>
        A standard <Code>{"C_{in} = 8"}</Code>, <Code>{"C_{out} = 8"}</Code>, <Code>k = 3</Code> conv has <Code>{"8 \\cdot 8 = 64"}</Code> independent 3×3 filters — one for every (output channel, input channel) pair. A depthwise conv at the same shape has exactly <Code>{"C_{in} = 8"}</Code> filters, each applied to only its matching input channel with no cross-channel mixing. The heatmap below shows which (output channel, input channel) pairs have their own learned filter: 1 means "this pair has an independent 3×3 kernel"; 0 means "the filter for this pair is tied to other pairs or is zero."
      </Prose>

      <Heatmap
        label="kernel-pair occupancy: standard conv (top) vs depthwise conv (bottom) at Cin=Cout=8"
        rowLabels={[
          "std out0", "std out1", "std out2", "std out3", "std out4", "std out5", "std out6", "std out7",
          "dw out0", "dw out1", "dw out2", "dw out3", "dw out4", "dw out5", "dw out6", "dw out7",
        ]}
        colLabels={["in0", "in1", "in2", "in3", "in4", "in5", "in6", "in7"]}
        matrix={[
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 1],
        ]}
        colorScale="gold"
      />

      <Prose>
        The top 8 rows (standard) are a dense all-ones block: 64 independent filters, 576 learned weights. The bottom 8 rows (depthwise) are a diagonal: 8 independent filters, 72 learned weights. The depthwise stage has no cross-channel interaction at all; a separate 1×1 pointwise conv provides the channel mixing and is the second half of the full depthwise separable. The two together have <Code>{"8 \\cdot 9 + 8 \\cdot 8 = 136"}</Code> weights versus <Code>576</Code> for the standard conv — 4.2× fewer at these small dimensions, approaching 9× asymptotically.
      </Prose>

      <H3>6b. Receptive field growth: linear vs exponential dilation</H3>

      <Prose>
        The plot compares receptive field size across a stack of 7 layers for two dilation schedules: all-ones (standard convs) grows linearly as <Code>{"1 + 2L"}</Code>; exponential <Code>{"d_\\ell = 2^{\\ell - 1}"}</Code> grows geometrically. The y-axis is log-scale — on the same axes, linear would look flat compared to the exponential curve.
      </Prose>

      <Plot
        label="receptive field vs depth, k=3 convs"
        xLabel="layer index"
        yLabel="receptive field (pixels)"
        series={[
          {
            name: "standard convs (d=1)",
            color: "#f87171",
            points: [
              [1, 3],
              [2, 5],
              [3, 7],
              [4, 9],
              [5, 11],
              [6, 13],
              [7, 15],
            ],
          },
          {
            name: "exponential dilation (d=1,2,4,...)",
            color: colors.gold,
            points: [
              [1, 3],
              [2, 7],
              [3, 15],
              [4, 31],
              [5, 63],
              [6, 127],
              [7, 255],
            ],
          },
        ]}
      />

      <Prose>
        By layer 7, standard convs cover a 15×15 window; exponential dilation covers 255×255 — larger than a full ImageNet image. The parameter counts are identical. This is the reason dilated convs became the default spatial operator for segmentation: the alternative was to pool and upsample, which loses resolution; dilated stacks preserve resolution and still see the whole image.
      </Prose>

      <H3>6c. MobileNetV2 inverted residual forward pass — StepTrace</H3>

      <StepTrace
        label="MobileNetV2 inverted residual block forward pass"
        steps={[
          {
            label: "Step 1 — Input arrives in narrow representation",
            render: () => (
              <Prose>
                {"Input x has shape [B, C, H, W] with C the 'narrow' or 'bottleneck' channel count (e.g., C=32 in an early MobileNetV2 stage). The block will compute y = project(dw(expand(x))) + x. The skip is held onto here; the expand/depthwise/project chain operates on copies."}
              </Prose>
            ),
          },
          {
            label: "Step 2 — 1x1 expand to high-dimensional space",
            render: () => (
              <Prose>
                {"The 1x1 expansion conv projects C channels to t*C channels (t=6 is the default). For C=32 this goes to 192. This is cheap: 1x1 conv on a narrow input costs C * tC * H * W = 32 * 192 * H * W = 6144 * HW FLOPs. BN then ReLU6 follow. The network is now 'wide.'"}
              </Prose>
            ),
          },
          {
            label: "Step 3 — 3x3 depthwise on wide tensor",
            render: () => (
              <Prose>
                {"The 3x3 depthwise operates at the wide channel count tC = 192, but because it is grouped (one 3x3 filter per channel, no mixing), the cost is 9 * tC * H * W = 1728 * HW FLOPs — cheap despite the high channel count. This is the whole point of inversion: depthwise's cost scales linearly in channels, so widening is free. BN + ReLU6 follow."}
              </Prose>
            ),
          },
          {
            label: "Step 4 — 1x1 linear projection back to narrow",
            render: () => (
              <Prose>
                {"The final 1x1 projects from tC = 192 back to C = 32 channels. Cost: tC * C * H * W = 6144 * HW FLOPs — symmetric with the expansion. Critically, NO activation follows — only BatchNorm. Sandler et al. 2018 argue that ReLU on a low-dimensional representation destroys information (negative activations are clipped, and the manifold in the narrow space cannot tolerate that loss). Keeping this projection linear is the 'linear bottleneck' from the paper's title."}
              </Prose>
            ),
          },
          {
            label: "Step 5 — Residual addition (only if shapes match)",
            render: () => (
              <Prose>
                {"y = project(...) + x, but only if stride == 1 AND input_channels == output_channels. For downsampling blocks (stride=2) or channel-change blocks, the skip is dropped and only the expand/dw/project path feeds forward. This makes the skip 'conditional' — present in most blocks of the network, absent at stage transitions."}
              </Prose>
            ),
          },
          {
            label: "Step 6 — Output in narrow representation",
            render: () => (
              <Prose>
                {"y has the same shape as x (for stride-1 blocks). The next block sees narrow-channel input. This is the 'inversion' vs ResNet: ResNet carries the WIDE representation in its skip path and does compute in the NARROW bottleneck; MobileNetV2 carries the NARROW representation in its skip path and does compute in the WIDE middle. Both save FLOPs; they just choose opposite regimes for which part of the stack is narrow."}
              </Prose>
            ),
          },
        ]}
      />

      <H3>6d. MobileNet family vs ResNet on ImageNet</H3>

      <Prose>
        The plot shows reported top-1 ImageNet accuracy against multiply-add count (in millions) for the MobileNet family and ResNet baselines. MobileNetV3-Large at ~220M MACs matches ResNet-50 at ~4100M MACs — a 19× FLOP reduction for the same accuracy. At the higher end, EfficientNet-B4 hits the same accuracy as ResNet-152 at 1/4 the FLOPs.
      </Prose>

      <Plot
        label="ImageNet top-1 accuracy vs MACs (lower-left = cheaper, upper-right = better)"
        xLabel="multiply-adds (millions)"
        yLabel="ImageNet top-1 accuracy (%)"
        series={[
          {
            name: "MobileNet family",
            color: colors.gold,
            points: [
              [150, 68.4],
              [300, 71.8],
              [60, 67.4],
              [220, 75.2],
              [390, 77.3],
            ],
          },
          {
            name: "ResNet family",
            color: "#f87171",
            points: [
              [1800, 70.4],
              [4100, 76.1],
              [7800, 77.4],
              [11300, 78.3],
            ],
          },
        ]}
      />

      <Prose>
        Every MobileNet point (v1-1.0, v2-1.0, v3-Small, v3-Large, EfficientNet-B0) sits to the upper-left of the ResNet Pareto frontier. The architectural factorizations pay for themselves: per FLOP, depthwise-separable with inverted-residual is roughly an order of magnitude more efficient at ImageNet accuracy than the standard ResNet block. The gap narrows at very high accuracy (past 82%), where attention-based architectures (ViT, Swin) take over, but for deployment below 80% top-1, the MobileNet family remains the default.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <StepTrace
        label="when to use depthwise separable vs dilated vs standard convs"
        steps={[
          {
            label: "Mobile / edge deployment — depthwise separable + inverted residual",
            render: () => (
              <Prose>
                {"If your target is a phone, embedded device, browser via WebAssembly, or any latency-sensitive endpoint with memory bandwidth limits, depthwise separable is the default. MobileNetV3, EfficientNet-Lite, and MobileViT all use it. Inverted residuals add the right amount of expressivity on top. For quantization (int8), depthwise separable blocks quantize especially well because the per-channel depthwise weights are naturally small-magnitude; standard convs often need per-channel scales just to survive 8-bit rounding."}
              </Prose>
            ),
          },
          {
            label: "Dense prediction (segmentation, depth, flow) — dilated convolutions",
            render: () => (
              <Prose>
                {"For semantic segmentation, instance segmentation, depth estimation, or optical flow, dilated convs are the standard spatial operator after the classifier backbone. DeepLab V3+, SegFormer (for its mix-transformer stages), and HRNet all use them. The alternative — encoder-decoder with upsampling — is also viable but loses resolution in the bottleneck. If you need per-pixel output at full input resolution, dilated convs let the backbone produce 1/8 or 1/16 resolution features with enormous RF, and ASPP provides multi-scale context without further downsampling."}
              </Prose>
            ),
          },
          {
            label: "Server-scale classification / detection — standard convs (or transformers)",
            render: () => (
              <Prose>
                {"When FLOPs are not the binding constraint, standard convs remain competitive. ResNet-50, ResNeXt, and RegNet-Y all beat MobileNet on per-parameter accuracy at server scale. By 2026, for new server-scale vision work, the question is usually convs vs transformers; within convs, the default is 'use depthwise separable unless you have a specific reason not to' because the accuracy loss is small and the compute savings compound at scale."}
              </Prose>
            ),
          },
          {
            label: "Intermediate efficiency — grouped convolutions with modest groups",
            render: () => (
              <Prose>
                {"If depthwise (groups = Cin) costs too much accuracy on your task and standard (groups = 1) is too expensive, a grouped conv with groups in 2-32 is a middle ground. ResNeXt uses groups = 32 at a typical block width; ShuffleNet uses groups = 4 plus channel shuffle to preserve cross-group information. The generalization is clean: depthwise is groups = Cin; standard is groups = 1; any intermediate is a valid efficiency-accuracy trade-off."}
              </Prose>
            ),
          },
          {
            label: "Transformers have taken over for many of these — but check",
            render: () => (
              <Prose>
                {"For tasks with ~1B parameters and enough data (billions of tokens or images), Vision Transformers beat CNNs on most benchmarks. For tasks with <100M parameters, limited data, or strict deployment constraints, CNNs with depthwise separable blocks still win — often by large margins (30-50% of a ViT's compute at the same accuracy for mobile-size models). The rule of thumb: if you're training a foundation vision model, use a ViT; if you're deploying to an edge device or a constrained task, use a modern CNN with depthwise separable blocks."}
              </Prose>
            ),
          },
          {
            label: "Avoid — naive stacked dilations, depthwise on very small channel counts",
            render: () => (
              <Prose>
                {"Never stack dilated convs at the same rate (gridding artifact — see section 9). Never use depthwise on very small channel counts (<32) without careful testing; the information bottleneck of depthwise on narrow tensors can be severe, which is why MobileNetV2's inverted residual expands first. And never use depthwise in the first stem layer of a network (Chollet 2017 explicitly notes this); the input image has only 3 channels and there's no cross-channel correlation to separate out."}
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales</H2>

      <H3>8.1 Parameter efficiency: MobileNet matches AlexNet at 1/14 the params</H3>

      <Prose>
        MobileNet-1.0 achieves 70.6% ImageNet top-1 with 4.2M parameters. AlexNet achieves 57.1% with 60M parameters. At comparable accuracy (~60%), MobileNet uses ~14× fewer parameters. The headline number is "depthwise separable reduces parameters by 8–9×," but the real gain in MobileNet comes from stacking this factorization with a modern training recipe (BatchNorm, good initialization, learning rate schedule) — the network's per-parameter accuracy is roughly 3× AlexNet's even before the depthwise-separable factor. At ResNet-50-like accuracy (~76%), MobileNet-V2 uses 3.4M params vs ResNet-50's 25.6M: a 7.5× reduction.
      </Prose>

      <H3>8.2 Depthwise conv underutilizes GPUs — memory-bandwidth-bound</H3>

      <Prose>
        Depthwise convolutions have an arithmetic intensity (FLOPs per byte of memory access) roughly <Code>k² / 2 = 4.5</Code> for <Code>k = 3</Code>. A standard conv has arithmetic intensity scaling with channels, typically 100+ on modern hardware. Modern GPUs deliver 10-30 FLOPs per byte of memory access at peak (the "roofline"); anything below that saturates memory bandwidth rather than compute. Depthwise at intensity 4.5 is firmly in the bandwidth-bound regime on GPUs — it runs at a fraction of peak FLOPs/s. This is why MobileNet shows only modest GPU speedups despite its 8× FLOP reduction. On mobile CPUs (Cortex-A77, Apple M-series E-cores), memory bandwidth is a smaller constraint and depthwise gives its full speedup. MobileNet's target hardware was never GPUs.
      </Prose>

      <H3>8.3 Fused conv-BN is critical for mobile inference</H3>

      <Prose>
        At inference time, BatchNorm can be folded into the preceding conv's weights:
      </Prose>

      <MathBlock>
        {"W_{\\text{fused}}[c, :, :, :] = \\frac{\\gamma_c}{\\sqrt{\\sigma_c^2 + \\epsilon}} \\cdot W[c, :, :, :], \\quad b_{\\text{fused}, c} = \\beta_c - \\frac{\\gamma_c \\mu_c}{\\sqrt{\\sigma_c^2 + \\epsilon}}"}
      </MathBlock>

      <Prose>
        After folding, the BN layer disappears entirely; the conv absorbs its affine parameters. This saves roughly 25% of the MobileNet inference latency on typical mobile SoCs (BN's per-channel mean/variance reads dominated the memory bandwidth of the small depthwise ops). PyTorch's <Code>torch.ao.quantization.fuse_modules</Code> and TFLite's conversion pipeline both do this automatically. For anyone exporting a MobileNet-family model to deploy, checking that conv-BN fusion happened is the single biggest latency win available.
      </Prose>

      <H3>8.4 Dilated convs enable segmentation at input resolution</H3>

      <Prose>
        The alternative to dilation in segmentation is encoder-decoder with upsampling (U-Net, FPN). At input 512×512, a U-Net bottleneck is typically 16×16; the output must go through 5 stages of upsampling, each of which loses information. A DeepLab with output stride 16 produces 32×32 features with full receptive-field coverage via dilated convs, then a single bilinear upsample to output. The dilated approach preserves more spatial detail at large outputs: on Cityscapes 2048×1024 images, DeepLab V3+ produces mIoU ~82 vs U-Net variants at ~78. The trade-off is memory: a 32×32 feature map at full receptive field takes more activation memory than a 16×16 bottleneck. For 4K+ imagery, hybrid approaches (dilated on mid-res features, bilinear upsample for the final stage) are now standard.
      </Prose>

      <H3>8.5 Depthwise-separable + NAS is the hyperparameter landscape</H3>

      <Prose>
        By the time MobileNetV3 shipped (2019), the block structure had too many knobs for humans to tune by hand: kernel size (3 or 5), expansion ratio (1, 3, 6, 8), squeeze-excitation (on or off), activation (ReLU6 or h-swish), stride, output channels. Each MobileNet-V3 block picks a configuration from this menu. The menu has <Code>{"\\sim 2 \\cdot 4 \\cdot 2 \\cdot 2 \\cdot 2 \\cdot O"}</Code> options per block and there are ~20 blocks, giving a search space of <Code>{"10^{50}"}</Code>+ architectures. NAS (specifically, MnasNet's RL-based search and later differentiable methods like ProxylessNAS) is now the standard tool. Every frontier mobile architecture since 2019 has been NAS-searched within a depthwise-separable inverted-residual template.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Swapping standard convs for depthwise-separable everywhere</H3>

      <Prose>
        The naive move — take a ResNet-50 and replace every 3×3 conv with a 3×3 depthwise + 1×1 pointwise — usually costs 2–4 points of top-1 accuracy. The information loss from restricted cross-channel mixing accumulates across the network. MobileNet works because the architecture is designed <em>around</em> depthwise-separable: expansion ratios, channel counts, and block structure are all tuned to the factorization. Fix: don't retrofit; use a mobile-style architecture (MobileNetV3, EfficientNet-Lite) that was designed for depthwise-separable from scratch. If you must retrofit, keep the first and last conv stages standard, or use a mix of standard and separable blocks.
      </Prose>

      <H3>9.2 Dilation gridding — equal stacked dilations</H3>

      <Prose>
        Stacking multiple dilated convs at the same rate creates a sparse sampling pattern. Pixels at off-grid positions are never sampled, and small features aligned with those positions disappear. Symptom: segmentation output has checkerboard artifacts; boundaries are jagged; small thin objects (power lines, street signs) are systematically misclassified. Fix: use HDC rates (1, 2, 3) or (1, 2, 5) or (2, 3, 5) — any sequence with GCD 1. Wang et al. 2018 give a sufficient condition: the "max distance" formula in section 3.5. In practice, following a simple "co-prime or near-co-prime" heuristic is enough.
      </Prose>

      <CodeBlock language="python">
{`# Bug: equal stacked dilations -> gridding
dilated_stack_bad = nn.Sequential(
    nn.Conv2d(64, 64, 3, dilation=2, padding=2),
    nn.Conv2d(64, 64, 3, dilation=2, padding=2),
    nn.Conv2d(64, 64, 3, dilation=2, padding=2),
)

# Fix: HDC rates with GCD 1
dilated_stack_ok = nn.Sequential(
    nn.Conv2d(64, 64, 3, dilation=1, padding=1),
    nn.Conv2d(64, 64, 3, dilation=2, padding=2),
    nn.Conv2d(64, 64, 3, dilation=3, padding=3),
)`}
      </CodeBlock>

      <H3>9.3 Wrong groups value</H3>

      <Prose>
        PyTorch requires <Code>{"C_{in}"}</Code> to be divisible by <Code>groups</Code> and <Code>{"C_{out}"}</Code> to be divisible by <Code>groups</Code>. Forgetting this gives a runtime error:
      </Prose>

      <CodeBlock language="python">
{`# Bug: Cin=96 is not divisible by groups=32
nn.Conv2d(96, 96, 3, padding=1, groups=32)
# RuntimeError: in_channels must be divisible by groups

# Fix: choose groups that divides Cin and Cout
nn.Conv2d(96, 96, 3, padding=1, groups=96)    # depthwise
nn.Conv2d(96, 96, 3, padding=1, groups=48)    # 2 channels per group
nn.Conv2d(96, 96, 3, padding=1, groups=32)    # error — 96 / 32 = 3 (ok!)

# Actually 96 / 32 = 3, which IS valid. More subtle bug:
nn.Conv2d(96, 128, 3, padding=1, groups=32)   # Cin=96 ok (96/32=3), but Cout=128/32=4. ok
nn.Conv2d(96, 130, 3, padding=1, groups=32)   # ERROR: 130 not divisible by 32`}
      </CodeBlock>

      <Prose>
        The most common real-world failure is changing channel widths during NAS or width-multiplier scaling and forgetting to check divisibility. MobileNet's width multiplier <Code>α</Code> rounds all channel counts to multiples of 8 precisely to avoid this. Fix: round channels to a multiple of your largest planned group count; <Code>α</Code>-scaled MobileNets use <Code>{"\\text{round}(C \\cdot \\alpha / 8) \\cdot 8"}</Code>.
      </Prose>

      <H3>9.4 Depthwise + BN + small batch — GPU slowness and statistics issues</H3>

      <Prose>
        Depthwise on modern GPUs is memory-bandwidth-bound, and BatchNorm's per-channel statistics computation is memory-bandwidth-bound too. At small batch sizes (<16), the two together can be slower than a standard conv with BN of equivalent shape, even though FLOPs are 8× lower. Additionally, depthwise + BN has a statistics problem: with only <Code>B × H × W</Code> samples per channel for BN, very small batches give noisy running statistics that hurt train/eval consistency. Fix: either (a) use <Code>GroupNorm</Code> instead of BN (no batch dependence), (b) use <Code>SyncBatchNorm</Code> across workers to effectively increase batch size, or (c) use batch sizes ≥ 32 per device during training. PyTorch's <Code>nn.BatchNorm2d</Code> has known performance issues with groups-heavy convs at small batch; cudnn's fused conv-BN kernels help but only at batch ≥ 32.
      </Prose>

      <H3>9.5 ASPP without normalization inside the branches</H3>

      <Prose>
        A subtle DeepLab-style failure: build ASPP without BatchNorm inside the dilated branches. Each branch then produces features at different magnitudes (dilated convs at large rates see larger, more heterogeneous receptive fields and produce activations with different statistics). The concatenation produces a tensor where the dynamic range varies by branch, and the final 1×1 projection has to learn to compensate. Training is unstable; loss oscillates. Fix: always put BN (or GroupNorm for small-batch segmentation) inside each ASPP branch. Reference: see the torchvision ASPP implementation in section 5.4 — every branch has BN after the conv and before ReLU. Chen et al. 2017 emphasize this in DeepLab V3's ablations; it's a 2-3 mIoU difference on PASCAL VOC.
      </Prose>

      <H3>9.6 Depthwise in the first stem — no cross-channel correlation to separate</H3>

      <Prose>
        The first conv in an image-classifier network takes 3-channel RGB input. Depthwise separable at this stage means: one 3×3 filter per RGB channel (so 27 weights) followed by a 1×1 that can only mix the three resulting channels. There is no "cross-channel correlation" to factor out at this stage — the three channels are color channels, and the network hasn't built feature maps yet. Chollet (Xception, 2017) explicitly keeps the first conv standard. Symptom: if you put depthwise in the stem, the network is effectively blind to color information (the 1×1 pointwise over 3 channels cannot encode rich spatial-color correlations). Fix: first layer is always a standard conv in MobileNet, EfficientNet, and Xception. Depthwise-separable starts from layer 2 onward.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Read in roughly chronological order to follow the development from Sifre's factorization through NAS-searched mobile architectures and the parallel dilated-conv / segmentation thread.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "Sifre 2014 — Rigid-motion scattering (first depthwise separable)",
            render: () => (
              <Prose>
                Sifre, L. (2014). "Rigid-motion scattering for image classification." PhD thesis, École Polytechnique, Paris. Available at www.di.ens.fr/data/publications/papers/phd_sifre.pdf. The first published appearance of depthwise-separable convolution, presented as a natural factorization of 2D convolution in the context of wavelet scattering networks. Sifre shows that separating spatial and channel mixing recovers ~95% of the joint operator's accuracy at ~1/9 the compute. The thesis is dense and mathematical; the practical relevance was not picked up for another three years, but every MobileNet-family paper cites this as the origin.
              </Prose>
            ),
          },
          {
            label: "Howard et al. 2017 — MobileNets (arXiv 1704.04861)",
            render: () => (
              <Prose>
                Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., and Adam, H. (2017). "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." arXiv:1704.04861. Available at arxiv.org/abs/1704.04861. The paper that productionized depthwise separable convolutions. Section 3.1 derives the FLOP formula <Code>{"1/N + 1/D_K^2"}</Code>; section 3.2 introduces the width multiplier α and resolution multiplier ρ; section 4 reports ImageNet and COCO numbers. Figure 3 shows the design space across α ∈ (0.25, 0.5, 0.75, 1.0). MobileNet-1.0 at 4.2M params achieves 70.6% ImageNet top-1. The single most-cited mobile architecture paper.
              </Prose>
            ),
          },
          {
            label: "Chollet 2017 — Xception (arXiv 1610.02357)",
            render: () => (
              <Prose>
                Chollet, F. (2017). "Xception: Deep Learning with Depthwise Separable Convolutions." arXiv:1610.02357. Published in CVPR 2017. Available at arxiv.org/abs/1610.02357. Philosophically the most important paper in this thread: it frames the Inception block as an intermediate point on a spectrum between standard conv (joint spatial + channel) and depthwise separable (factored). Xception takes the extreme position (all depthwise separable) and shows it beats Inception-V3 on ImageNet at comparable FLOPs. The architecture is a deep stack of Separable-Conv → BN → ReLU blocks with residual connections (note: despite the name "Extreme Inception," Xception uses ResNet-style skips). Still one of the cleanest per-FLOP ImageNet architectures.
              </Prose>
            ),
          },
          {
            label: "Sandler et al. 2018 — MobileNetV2 (arXiv 1801.04381)",
            render: () => (
              <Prose>
                Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., and Chen, L.-C. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." arXiv:1801.04381. Published in CVPR 2018. Available at arxiv.org/abs/1801.04381. Introduces two design changes: inverted residuals (expand → depthwise → project) and linear bottlenecks (no activation on the final projection). Section 3 derives the information-theoretic argument for linear bottlenecks: the "manifold of interest" in the narrow representation is low-dimensional; ReLU on narrow representations collapses it. MobileNetV2-1.0 hits 72.0% at 3.4M params. The inverted residual is now the default block for nearly every efficient architecture (EfficientNet, MobileViT, MobileNeXt).
              </Prose>
            ),
          },
          {
            label: "Howard et al. 2019 — MobileNetV3 (arXiv 1905.02244)",
            render: () => (
              <Prose>
                Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q.V., and Adam, H. (2019). "Searching for MobileNetV3." arXiv:1905.02244. Published in ICCV 2019. Available at arxiv.org/abs/1905.02244. The first MobileNet where the block configuration was selected by Neural Architecture Search on a latency-aware reward rather than by human design. Introduces h-swish (a piecewise-linear swish approximation), squeeze-and-excitation inside inverted residuals, and NetAdapt-based fine-tuning. MobileNetV3-Large achieves 75.2% top-1 at 5.4M params with measured Pixel-1 latency. The paper established the NAS + depthwise-separable template that all subsequent mobile architectures follow.
              </Prose>
            ),
          },
          {
            label: "Tan & Le 2019 — EfficientNet (arXiv 1905.11946)",
            render: () => (
              <Prose>
                Tan, M. and Le, Q.V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." arXiv:1905.11946. Published in ICML 2019. Available at arxiv.org/abs/1905.11946. The paper that systematically studied how to scale depth, width, and input resolution jointly rather than independently. Starting from MobileNet-V2's inverted residual template, NAS produces EfficientNet-B0; compound scaling then generates B1-B7. EfficientNet-B7 achieves 84.3% top-1 with 66M params — at the time, state of the art on ImageNet. The compound scaling formula <Code>{"d = \\alpha^\\phi, w = \\beta^\\phi, r = \\gamma^\\phi"}</Code> (constrained <Code>{"\\alpha \\cdot \\beta^2 \\cdot \\gamma^2 \\approx 2"}</Code>) is a key reference for any model-scaling work.
              </Prose>
            ),
          },
          {
            label: "Yu & Koltun 2016 — Dilated convolutions (arXiv 1511.07122)",
            render: () => (
              <Prose>
                Yu, F. and Koltun, V. (2016). "Multi-Scale Context Aggregation by Dilated Convolutions." arXiv:1511.07122. Published in ICLR 2016. Available at arxiv.org/abs/1511.07122. The paper that introduced dilated (or "atrous") convolutions to deep learning for semantic segmentation. Section 2 defines the operator; section 3 builds a dilation-only context module that plugs into existing segmentation networks and improves PASCAL VOC 2012 mIoU by ~4 points. The experimental setup established dilation as the tool for preserving resolution in dense prediction. The term "atrous" comes from French for "with holes" (à trous) and was independently used in earlier signal processing work (Holschneider 1989, à trous wavelet transform).
              </Prose>
            ),
          },
          {
            label: "Chen et al. 2017 — DeepLab (arXiv 1606.00915)",
            render: () => (
              <Prose>
                Chen, L.-C., Papandreou, G., Kokkinos, I., Murphy, K., and Yuille, A.L. (2017). "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs." arXiv:1606.00915. Published in TPAMI 2018. Available at arxiv.org/abs/1606.00915. Section 3.1 introduces ASPP (Atrous Spatial Pyramid Pooling) with dilated convolutions at rates 6, 12, 18, 24. Section 4 reports PASCAL VOC 2012 mIoU = 79.7% (then state of the art). DeepLab V3 (arXiv 1706.05587) refined the architecture to drop the CRF, and DeepLab V3+ (arXiv 1802.02611) added a lightweight decoder. This paper series defines the segmentation baselines every subsequent paper must beat, and ASPP is still in use in 2026.
              </Prose>
            ),
          },
          {
            label: "Wang et al. 2018 — Understanding Convolution for Semantic Segmentation (gridding)",
            render: () => (
              <Prose>
                Wang, P., Chen, P., Yuan, Y., Liu, D., Huang, Z., Hou, X., and Cottrell, G. (2018). "Understanding Convolution for Semantic Segmentation." arXiv:1702.08502. Published in WACV 2018. Available at arxiv.org/abs/1702.08502. Section 3 identifies the gridding artifact: when several dilated convs are stacked at the same rate, the effective sampling pattern is a sparse grid that misses most pixels in the nominal receptive field. Section 4 proposes Hybrid Dilated Convolution (HDC): use rates with GCD 1 so consecutive layers cover each other's gaps. The paper proves a sufficient condition on the dilation sequence for full coverage. Every production segmentation architecture since 2018 uses HDC-style rate selection, even when it's not explicitly named.
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
        Attempt all five before reading the answers. Exercises 1–2 test the math; 3 tests the intuition behind inverted residuals; 4 tests segmentation architecture judgment; 5 tests debugging.
      </Prose>

      <H3>Exercise 1 (FLOP derivation)</H3>
      <Prose>
        A standard 3×3 conv with <Code>{"C_{in} = 256"}</Code> and <Code>{"C_{out} = 256"}</Code> operates on a 14×14 feature map. Compute (a) the parameter count, (b) the FLOP count, (c) the equivalent depthwise-separable cost, (d) the ratio. Then extend to <Code>{"C_{in} = 512, C_{out} = 1024"}</Code>. What does the ratio approach as <Code>{"C_{out} \\to \\infty"}</Code>?
      </Prose>
      <Callout accent="green">
        <strong>Answer 1.</strong> (a) <Code>{"9 \\cdot 256 \\cdot 256 = 589{,}824"}</Code> params. (b) <Code>{"589{,}824 \\cdot 14 \\cdot 14 = 115{,}605{,}504"}</Code> FLOPs. (c) Depthwise-separable: <Code>{"9 \\cdot 256 + 256 \\cdot 256 = 2304 + 65{,}536 = 67{,}840"}</Code> params; <Code>{"67{,}840 \\cdot 14 \\cdot 14 = 13{,}296{,}640"}</Code> FLOPs. (d) Ratio <Code>{"115.6M / 13.3M \\approx 8.70\\times"}</Code>. For <Code>{"C_{in}=512, C_{out}=1024"}</Code>: standard = <Code>{"9 \\cdot 512 \\cdot 1024 = 4{,}718{,}592"}</Code> params, separable = <Code>{"9 \\cdot 512 + 512 \\cdot 1024 = 528{,}896"}</Code> params, ratio <Code>{"\\approx 8.92\\times"}</Code>. As <Code>{"C_{out} \\to \\infty"}</Code>, <Code>{"1/C_{out} \\to 0"}</Code> and the ratio approaches <Code>{"1 / (1/k^2) = k^2 = 9"}</Code> for <Code>k = 3</Code>. The 9× savings is the asymptotic limit — you can never save more than <Code>{"k^2"}</Code>× with this factorization because the depthwise cost itself is <Code>{"k^2 / C_{out}"}</Code> the standard cost.
      </Callout>

      <H3>Exercise 2 (receptive field — dilation schedule)</H3>
      <Prose>
        You have a segmentation backbone with 8 layers of 3×3 convs. What dilation schedule maximizes receptive field while avoiding the gridding artifact? Give the schedule and compute the resulting RF. Why is all-ones dilation bad? Why is all-twos dilation bad?
      </Prose>
      <Callout accent="green">
        <strong>Answer 2.</strong> All-ones (standard convs): <Code>{"\\text{RF} = 1 + 8 \\cdot 2 = 17"}</Code> pixels — too small for segmentation on a 512×512 image. All-twos: gridding artifact — stacking 8 convs at <Code>d=2</Code> produces an effective sampling pattern that only covers the "even-indexed" lattice within the receptive field, missing half the pixels. HDC-compliant schedule: use co-prime or near-co-prime rates, e.g., <Code>{"(1, 2, 3, 1, 2, 3, 1, 2)"}</Code> or (for geometric growth while avoiding gridding) <Code>{"(1, 2, 5, 1, 2, 5, 1, 2)"}</Code>. For the first schedule: <Code>{"\\text{RF} = 1 + 2 \\cdot (1 + 2 + 3 + 1 + 2 + 3 + 1 + 2) = 1 + 2 \\cdot 15 = 31"}</Code>. For exponential + reset <Code>{"(1, 2, 4, 8, 1, 2, 4, 8)"}</Code> (each group of 4 co-prime with each other): <Code>{"\\text{RF} = 1 + 2 \\cdot 2 \\cdot (1 + 2 + 4 + 8) = 1 + 4 \\cdot 15 = 61"}</Code>. The reset pattern (periodic dilation) is DeepLab's standard for stages: rates 1-2-5 per block within each stage.
      </Callout>

      <H3>Exercise 3 (inverted residual vs ResNet bottleneck)</H3>
      <Prose>
        Sandler et al. 2018 argue that the inverted residual is dual to ResNet's bottleneck. Explain (a) what each block's "narrow" and "wide" states are, (b) why MobileNetV2 chose the wide state as the middle rather than the ends, (c) what goes wrong if you apply ReLU to the final 1×1 projection (the "linear bottleneck" argument).
      </Prose>
      <Callout accent="green">
        <strong>Answer 3.</strong> (a) ResNet bottleneck: input/output WIDE (e.g., 256 channels), middle NARROW (64). MobileNetV2 inverted residual: input/output NARROW (e.g., 32 channels), middle WIDE (192). (b) Depthwise 3×3 has cost <Code>{"9 \\cdot C_m \\cdot H \\cdot W"}</Code> which is LINEAR in channel count, so running depthwise on the wide middle is cheap. The expensive parts are the 1×1 projections, whose cost scales as <Code>{"C_{\\text{narrow}} \\cdot C_{\\text{wide}}"}</Code>. MobileNetV2 keeps <Code>{"C_{\\text{narrow}}"}</Code> small, so the 1×1s are cheap too. In a ResNet bottleneck, the 3×3 middle is the expensive standard conv with cost <Code>{"9 \\cdot C_{\\text{narrow}}^2"}</Code>, which is cheap because NARROW is small. Both are bottleneck designs; they just choose opposite widths at opposite points. (c) Linear bottleneck argument: the "manifold of interest" in the narrow representation is low-dimensional (the network has already compressed the useful information). ReLU clips negative activations to zero. When applied to a low-dimensional manifold, this clipping destroys information — the manifold had to have some negative-valued projections to distinguish classes, and ReLU removes them. At high dimensions (the expanded state), there is enough redundancy that ReLU is harmless. Empirically, removing the final ReLU improves ImageNet top-1 by ~1% — a very large gain for a one-line change.
      </Callout>

      <H3>Exercise 4 (ASPP design)</H3>
      <Prose>
        You are designing a segmentation head for a backbone that produces features at output stride 16 (i.e., feature map is 1/16 the input resolution). The input images are 512×512, so feature maps are 32×32. What dilation rates should you use in ASPP? If you switched to output stride 8 (64×64 features), how should the rates change? Why?
      </Prose>
      <Callout accent="green">
        <strong>Answer 4.</strong> The point of ASPP is to sample context at a range of INPUT-pixel scales regardless of the feature resolution. For output stride 16: each feature-map pixel corresponds to a 16×16 block of input pixels. A dilated 3×3 conv at rate <Code>r</Code> on the feature map has effective receptive field <Code>{"(3-1) \\cdot r + 1 = 2r + 1"}</Code> feature-map pixels, which is <Code>{"16 \\cdot (2r+1)"}</Code> input pixels. DeepLab's standard rates (6, 12, 18) give effective input-pixel RFs of 208, 400, 592 — sampling input context at ~1/3, 2/3, and ~full image scale. For output stride 8: each feature pixel corresponds to 8 input pixels, so to keep the same input-pixel scales we double the dilation rates to (12, 24, 36). Chen et al. 2017 explicitly do this; their Table 4 shows the doubling yields the best PASCAL VOC mIoU at output stride 8. General rule: rate × output_stride should be approximately constant across stride choices, so that ASPP sees the same real-world scale regardless of how much the backbone has downsampled.
      </Callout>

      <H3>Exercise 5 (debugging — latency regression)</H3>
      <Prose>
        You port a ResNet-50 image classifier to MobileNetV2-1.0 to reduce on-device latency. On your CPU benchmark, MobileNetV2's latency is 1.2× the ResNet-50 latency — slower despite 7× fewer parameters and 7× fewer FLOPs. What are three likely causes, and how do you verify each?
      </Prose>
      <Callout accent="green">
        <strong>Answer 5.</strong> Three plausible causes, all consistent with "FLOPs went down but latency went up":
        <br />
        (1) <strong>Conv-BN not fused.</strong> At inference, BatchNorm should be folded into the preceding conv's weights (see section 8.3). If it isn't, BN's per-channel reads dominate the small depthwise ops' memory bandwidth and latency blows up. Verify: count the number of ops in the exported model (e.g., print the ONNX graph). If you see separate Conv and BN nodes, fusion didn't happen. Fix: call <Code>torch.ao.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])</Code> before export, or use ONNX simplifier's constant folding.
        <br />
        (2) <strong>Running on a backend where depthwise is slow.</strong> Some inference backends have great standard-conv kernels (cuDNN, MKLDNN) but mediocre depthwise kernels. On server CPUs with AVX-512, standard convs hit peak throughput; depthwise is often unoptimized and falls back to a generic im2col loop. Verify: profile with <Code>torch.profiler</Code> or per-op timing; if depthwise conv is 50%+ of latency despite being 15% of FLOPs, you're on a bad backend. Fix: switch to NNPACK, XNNPACK, or TFLite for mobile-oriented architectures.
        <br />
        (3) <strong>Too many small kernels — launch overhead.</strong> MobileNetV2 has more than 50 conv layers vs ResNet-50's ~50 (similar count) but each MobileNet op is much smaller (less compute). On a GPU or backend with per-kernel launch overhead, the MobileNet graph takes longer to execute even if each op is individually faster. Verify: measure kernel count and per-kernel time; if most ops run in <50µs and you have 150+ of them, launch overhead is bottlenecking. Fix: fuse ops (expand+depthwise+project into one kernel, as TFLite's "FusedBatchNorm" + "DepthwiseConv2D" does), or use graph-mode inference (torchscript, ONNX Runtime).
      </Callout>

    </div>
  ),
};

export default depthwiseDilatedContent;
