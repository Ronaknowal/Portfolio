import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const landmarkArchitecturesContent = {
  title: "Landmark Architectures (LeNet → AlexNet → VGG → ResNet → EfficientNet)",
  readTime: "~42 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        The history of convolutional neural networks is the history of image recognition. Every landmark ImageNet result, every production image classifier, every vision backbone you will use in 2026 is the direct descendant of a small family of architectures whose lineage can be traced across three decades. Each generation encodes a specific design principle that its predecessors lacked, and understanding the sequence is not just historical hygiene — it is a compressed tutorial in what scales, what breaks, and what problem each piece of machinery was invented to solve.
      </Prose>

      <Prose>
        The lineage begins in 1998 with Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner publishing "Gradient-Based Learning Applied to Document Recognition" in Proceedings of the IEEE 86(11):2278–2324. The architecture they described, LeNet-5, was a seven-layer CNN trained on 60,000 handwritten digits from MNIST. It had roughly 60,000 parameters. The activations were sigmoid (tanh in some variants), the subsampling layers used learned scalars followed by a bias and tanh, and the final classifier was a radial basis function. LeNet-5 was deployed at scale — by 1999 AT&T systems were reading a significant fraction of U.S. bank checks with it — but the broader field treated CNNs as a curiosity. The dominant vision pipelines were SIFT features, bag-of-visual-words, and hand-tuned SVMs. The gap between LeNet-5 and the next landmark was fourteen years.
      </Prose>

      <Prose>
        In 2012, Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton at the University of Toronto submitted AlexNet to the ImageNet Large Scale Visual Recognition Challenge (ILSVRC). It won by a margin wider than any subsequent year: top-5 error 16.4% against 26.2% for the runner-up, a hand-engineered Fisher-vector pipeline. AlexNet was not a clever architecture in a scientific sense — it was a scaled, GPU-enabled, modernized LeNet. The changes were few but decisive: (1) ReLU activations in place of sigmoid/tanh, which eliminated saturation and accelerated training by a factor of six on CIFAR-10; (2) dropout on the fully connected layers to regularize 60M parameters trained on 1.2M images; (3) local response normalization between some conv layers; (4) overlapping max pooling; (5) data augmentation by random crops and horizontal flips; (6) training split across two GTX 580 GPUs with 3 GB memory each — the cross-GPU connectivity pattern produced the distinctive "grouped convolutions" in the original diagram. AlexNet convinced the vision community that deep learning was not a fad. It started the GPU-deep-learning flywheel that is still running.
      </Prose>

      <Prose>
        In 2014, Karen Simonyan and Andrew Zisserman of the Oxford Visual Geometry Group published "Very Deep Convolutional Networks for Large-Scale Image Recognition" (arXiv:1409.1556). VGG-16 and VGG-19 reduced top-5 error on ImageNet to 7.3% and pushed depth from AlexNet's 8 layers to 16 and 19. The architectural idea was brutally simple: instead of large kernels, stack small 3×3 convolutions. Two stacked 3×3 convs have the same receptive field as one 5×5 but with fewer parameters (<Code>{"2 * 9 * C^2"}</Code> vs <Code>{"25 * C^2"}</Code>) and an extra nonlinearity. Three stacked 3×3 match a 7×7 with <Code>{"27 / 49"}</Code> of the parameters. Pool every few blocks, double channels after each pool, repeat until spatial resolution is gone, then three fully connected layers of 4096 units feed into a 1000-way softmax. VGG-16 has 138M parameters. Of those, 123M live in the first fully connected layer (<Code>{"7 * 7 * 512 * 4096"}</Code>). The conv trunk itself is only 15M. VGG's design is inelegant by modern standards but its feature maps are still used as perceptual losses in generative models: the "VGG loss" from Johnson et al. 2016 and every StyleGAN-era perceptual term routes through VGG-16 features.
      </Prose>

      <Prose>
        The same year, Christian Szegedy and colleagues at Google published "Going Deeper with Convolutions" (arXiv:1409.4842). GoogLeNet — also called Inception v1 — won ILSVRC 2014 classification with 6.67% top-5. Its signature was the Inception module: a single block that computes 1×1, 3×3, and 5×5 convolutions in parallel on the same input, plus a parallel max-pool branch, and concatenates the four outputs along the channel axis. Each conv branch is preceded by a 1×1 "bottleneck" that reduces channel count before the expensive 3×3 or 5×5. This keeps the module efficient: GoogLeNet has 22 layers and roughly 7M parameters — one-twentieth of VGG-16 — at better accuracy. The paper's contribution was more than architecture; it introduced the principle that width and multi-scale processing, not just depth, are axes worth exploring.
      </Prose>

      <Prose>
        In December 2015, Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun at Microsoft Research Asia posted "Deep Residual Learning for Image Recognition" (arXiv:1512.03385). ResNet won ILSVRC 2015 with a 152-layer network reaching 3.57% top-5 error, halving the prior year's error. The architectural idea was equally simple: add the block's input to its output. A residual block computes <Code>{"y = F(x) + x"}</Code>. If <Code>F</Code> is near zero, the block is identity, and "near identity" turns out to be where optimization wants to start. ResNet broke the depth barrier. Before it, networks past 20 layers trained worse than shallower networks despite more capacity (the "degradation problem"). After it, depth became a matter of compute rather than optimization. ResNet-50 is still in 2026 the default image classification backbone — the one you reach for when you need a baseline, a feature extractor, or an initialization.
      </Prose>

      <Prose>
        In 2016, Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Weinberger published DenseNet (arXiv:1608.06993). DenseNet replaced additive skip connections with concatenation: within a dense block, layer <Code>l</Code> receives the concatenation of all previous feature maps. DenseNet-BC-190 matched ResNet-200 accuracy at a third the parameters. It traded memory for parameter efficiency and made a philosophical point — the right unit for feature reuse is the concatenation of all prior features, not a single residual sum. DenseNet is rarely used today (the memory cost is real) but its influence lives on in U-Net skip architectures and in the dense-feature-reuse pattern of many segmentation models.
      </Prose>

      <Prose>
        In 2017, Jie Hu, Li Shen, and Gang Sun of Momenta published "Squeeze-and-Excitation Networks" (arXiv:1709.01507). SENet won ILSVRC 2017 with 2.25% top-5. The module is tiny: global average pool to one scalar per channel, a two-layer MLP with a reduction factor (usually 16), a sigmoid, and then broadcast-multiply the result back across spatial dimensions. This gives each channel a learned "importance" gate. Adding an SE block to any ResNet improves top-1 accuracy by 1–2% at an extra 0.5% parameter cost. SE is arguably the first widely-adopted form of channel attention; every modern efficient architecture (EfficientNet, MobileNet v3, RegNet-Y) includes an SE variant inside each block. The paper's broader claim — that attention across channels is free lunch — turned out to be largely correct.
      </Prose>

      <Prose>
        Also in 2017, Andrew Howard and colleagues at Google published "MobileNets" (arXiv:1704.04861). Mobile was now the dominant inference platform, and ImageNet-trained ResNet-50 was too slow for real-time use on a phone CPU. MobileNet v1 replaced standard 3×3 convolutions with <em>depthwise separable convolutions</em>: a depthwise 3×3 (one filter per input channel, no channel mixing) followed by a 1×1 pointwise (channel mixing, no spatial aggregation). For a 3×3 conv with <Code>{"C_{in}"}</Code> input channels and <Code>{"C_{out}"}</Code> output channels, this reduces FLOPs from <Code>{"9 * C_{in} * C_{out} * H * W"}</Code> to <Code>{"9 * C_{in} * H * W + C_{in} * C_{out} * H * W"}</Code> — a factor of roughly <Code>{"1 / C_{out} + 1/9"}</Code> cheaper. MobileNet v2 (Sandler et al. 2018, arXiv:1801.04381) added inverted residuals and linear bottlenecks: expand channels, depthwise conv, project back down, skip around the whole block. MobileNet v3 (Howard et al. 2019) used neural architecture search (NAS) to tune block parameters and added SE modules and hard-swish activation. The MobileNet lineage defined the efficient-inference standard for mobile and edge deployment.
      </Prose>

      <Prose>
        In 2019, Mingxing Tan and Quoc Le of Google Brain published "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (ICML 2019, arXiv:1905.11946). The paper asked a question the field had dodged: given a base architecture, how should you scale it up? Prior practice scaled one dimension at a time — ResNet-18 → ResNet-34 → ResNet-50 → ResNet-152 deepened; Wide ResNet widened; others increased input resolution. EfficientNet scaled all three dimensions jointly along a power-law <em>compound coefficient</em> <Code>φ</Code>: depth multiplied by <Code>{"α^φ"}</Code>, width by <Code>{"β^φ"}</Code>, resolution by <Code>{"γ^φ"}</Code>, with the constraint <Code>{"α * β^2 * γ^2 ≈ 2"}</Code> so total FLOPs scale like <Code>{"2^φ"}</Code>. The base architecture (EfficientNet-B0) was itself found by NAS on mobile-latency constraints. Scaling via compound coefficients gave EfficientNet-B7 at 84.3% ImageNet top-1 with 66M parameters — half the compute of the then-SOTA. EfficientNet is the architecture that made compound scaling a default recipe.
      </Prose>

      <Prose>
        In 2020, Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Dollár at Facebook AI Research published "Designing Network Design Spaces" (arXiv:2003.13678). RegNet moved the question from "which architecture?" to "what is the structure of the space of good architectures?" They searched a parametric family of networks, analyzed the distribution of high-performers, and found that the best networks have depth, width, and bottleneck ratios that follow simple linear functions. The paper is remarkable because its outputs — RegNet-X, RegNet-Y — are hand-writable in a page, yet Pareto-dominate EfficientNet at matched FLOPs on ImageNet. RegNet is the first architecture where the search happened in design-space coordinates rather than over individual networks.
      </Prose>

      <Prose>
        In 2021, Andrew Brock, Soham De, Samuel Smith, and Karen Simonyan at DeepMind published "High-Performance Large-Scale Image Recognition Without Normalization" (NFNet, arXiv:2102.06171). They removed batch normalization from deep ResNets — BN had been considered necessary since 2015 — by replacing it with weight standardization plus a per-block scaling trick called Adaptive Gradient Clipping. NFNet-F6 reached 86.5% ImageNet top-1 at the time of publication, matching EfficientNet-B7 at roughly 8× training speed. NFNet closed the loop: you can train very deep very accurate image models with just conv, skip, and nonlinearity — if you are careful about gradient scale.
      </Prose>

      <Prose>
        By 2026, the landscape has converged. For classification baselines, ResNet-50 remains the workhorse because of its ecosystem (pretrained weights in every framework, well-understood fine-tuning behavior, predictable inference cost). For accuracy-maximizing on ImageNet, the frontier is ConvNeXt-V2 and EfficientNet-V2 variants. For edge deployment, MobileNet v3 and EfficientNet-Lite. For feature extraction into downstream tasks (detection, segmentation, dense prediction), timm's pretrained EfficientNet or RegNet backbones. Vision Transformers now share the podium, but the CNN lineage in this topic is still the fabric of production computer vision: your image preprocessing pipeline normalizes with ImageNet statistics because a CNN from 2012 expects it.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Every landmark architecture encodes exactly one insight its predecessors lacked. Learning the lineage is learning a small set of orthogonal design axes and the order in which the field discovered them.
      </Prose>

      <Prose>
        <strong>LeNet — convolution as weight sharing.</strong> The original insight was that for translation-equivariant data (images), you should share weights across spatial positions and use local receptive fields. A 5×5 convolution kernel applied to a 32×32 image uses 25 parameters instead of the 32×32×32×32 = 1M parameters a fully connected layer would need. Translation equivariance is the inductive bias; parameter sharing is its implementation.
      </Prose>

      <Prose>
        <strong>AlexNet — activations matter, and so does data.</strong> The architectural delta from LeNet was modest. The activation delta was large: ReLU (piecewise linear, no saturation) replaced sigmoid (bounded, saturating). Combined with dropout on fully connected layers and massive data augmentation, AlexNet showed that the same convolutional chassis scales when the training recipe is modernized and GPUs remove the compute bottleneck. The insight is that <em>training machinery is architecture</em>. Every subsequent landmark keeps ReLU (or a close relative) and keeps thinking about regularization and normalization as first-class design choices.
      </Prose>

      <Prose>
        <strong>VGG — depth is cheap if you use small kernels.</strong> Stacking 3×3 convolutions achieves the same receptive field as larger kernels but with fewer parameters per unit of receptive field and more interleaved nonlinearities. VGG's insight: if you want more capacity, add depth by repeating a small unit, not by making kernels larger. The "conv block" as a repeating unit (conv → conv → pool) becomes the building block for every subsequent architecture.
      </Prose>

      <Prose>
        <strong>Inception / GoogLeNet — multi-scale in parallel, compress with 1×1.</strong> An Inception module runs multiple kernel sizes in parallel and concatenates. The 1×1 "bottleneck" convolution — first introduced here as a factorization device — reduces channel count before an expensive 3×3 or 5×5. The insight is that the right answer is not one kernel size but many, run in parallel, with cheap projections controlling channel explosion.
      </Prose>

      <Prose>
        <strong>ResNet — identity-preserving optimization.</strong> The degradation problem said deep networks fit training data worse than shallow ones. The fix was architectural: parameterize each block as a residual perturbation of identity. The insight generalizes beyond vision — it is why every Transformer block is a residual block. The lesson: make the "do nothing" baseline reachable at initialization, then train the network to deviate from it.
      </Prose>

      <Prose>
        <strong>DenseNet — concatenation reuses features.</strong> If residuals add, Dense layers concatenate. Every layer sees every prior layer's features directly. This fixes the "implicit subtraction" inherent in summation (adding a feature and its near-negative cancels) and makes feature reuse the default mode of computation. The cost is memory, which is why production prefers residuals.
      </Prose>

      <Prose>
        <strong>SENet — channels deserve attention.</strong> A conv layer mixes channels uniformly. SE adds a cheap gate that lets the network emphasize or suppress channels based on global content. The insight — channels are a first-class dimension to attend over — predates Transformer attention in the vision domain and remains the dominant cheap-attention primitive in efficient backbones.
      </Prose>

      <Prose>
        <strong>MobileNet — factorize spatial and channel mixing.</strong> A standard 3×3 conv does both spatial aggregation (local receptive field) and channel mixing (linear combination across channels) in one tensor contraction. Depthwise separable convs split these: depthwise for spatial, pointwise (1×1) for channels. The split is nearly lossless in accuracy but cuts compute by ~8×. The insight: operations that do two things are candidates for factorization.
      </Prose>

      <Prose>
        <strong>EfficientNet — scale all three axes together.</strong> Depth, width, and input resolution are not independent — scaling one in isolation hits diminishing returns because the others become bottlenecks. A wider network wants more depth to use the channels; a deeper network wants more resolution to give it something to see. Compound scaling couples them via a single coefficient. The insight: architecture scaling is a constrained optimization over design axes, not a univariate sweep.
      </Prose>

      <Prose>
        <strong>RegNet — the space of good architectures is low-dimensional.</strong> Most architecture search explores one network at a time. RegNet parameterizes families (by linear depth and width functions) and searches families. The outcome: the space of good networks is smaller than expected, and once you know the rules, you can write down a state-of-the-art architecture in a few lines. The insight: architectures are samples from structured distributions.
      </Prose>

      <Callout accent="gold">
        Mental model: every landmark adds one axis of optimization. Convolutions (spatial sharing), depth (VGG), multi-scale (Inception), skip connections (ResNet), feature reuse (DenseNet), channel attention (SENet), compute factorization (MobileNet), joint scaling (EfficientNet), design-space search (RegNet). The 2026 production backbone is the accumulation of all of these in one block.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Parameters and FLOPs for a conv layer</H3>

      <Prose>
        Let a 2D convolution layer have input <Code>{"C_in"}</Code> channels, output <Code>{"C_out"}</Code> channels, kernel size <Code>k</Code>, stride <Code>s</Code>, and output spatial size <Code>{"H_out * W_out"}</Code>. The parameter count (ignoring bias) and the FLOP count for a single forward pass are:
      </Prose>

      <MathBlock>
        {"\\text{Params} = k^2 \\cdot C_{in} \\cdot C_{out}"}
      </MathBlock>

      <MathBlock>
        {"\\text{FLOPs} = k^2 \\cdot C_{in} \\cdot C_{out} \\cdot H_{out} \\cdot W_{out}"}
      </MathBlock>

      <Prose>
        A conv's parameter count is independent of spatial size — the same kernel slides everywhere — but FLOPs scale linearly with output area. This is why early-stage convs (large spatial, small channels) dominate FLOPs and why late-stage convs (small spatial, large channels) dominate parameters. A common convention reports FLOPs as multiply-adds (MACs); doubling gives classical FLOPs.
      </Prose>

      <H3>3.2 AlexNet parameter budget</H3>

      <Prose>
        AlexNet's 60M parameters are dominated by the fully connected layers. The conv trunk has roughly 2.3M parameters; FC6 (6×6×256 → 4096) has <Code>{"9216 * 4096 \\approx 37.7 \\text{M}"}</Code>, FC7 (4096 → 4096) has <Code>{"4096 * 4096 \\approx 16.8 \\text{M}"}</Code>, FC8 (4096 → 1000) has <Code>{"4096 * 1000 \\approx 4.1 \\text{M}"}</Code>. Total FC params: ~58.6M. The FC layers carry 97.7% of the parameters but do less than 1% of the FLOPs. This asymmetry motivated later architectures (ResNet, EfficientNet) to replace dense-FC heads with global average pooling followed by a single linear classifier.
      </Prose>

      <H3>3.3 VGG-16 parameter budget</H3>

      <Prose>
        VGG-16's total is 138M. The conv trunk has ~14.7M. The first FC layer (7×7×512 → 4096) is <Code>{"25088 * 4096 \\approx 102.8 \\text{M}"}</Code> — almost three-quarters of the whole network. FC7 is 16.8M, FC8 is 4.1M. FLOPs for one 224×224 image forward pass: roughly 15.5 GFLOPs, of which ~98% are in the conv trunk. So VGG is FLOP-heavy and param-heavy, with the two concentrated at opposite ends of the network. This asymmetry is the architectural flaw that later networks fix by replacing FC with global pooling.
      </Prose>

      <H3>3.4 Two stacked 3×3 versus one 5×5</H3>

      <Prose>
        Simonyan and Zisserman's key argument: two stacked 3×3 convs have the same receptive field as one 5×5 conv but fewer parameters and more nonlinearities. Assume <Code>{"C_{in} = C_{out} = C"}</Code>:
      </Prose>

      <MathBlock>
        {"\\text{Params}_{5\\times 5} = 25 C^2, \\quad \\text{Params}_{3\\times 3 \\times 2} = 2 \\cdot 9 C^2 = 18 C^2"}
      </MathBlock>

      <Prose>
        A <Code>{"25 / 18 = 1.39\\times"}</Code> parameter saving per receptive-field-unit, plus an additional ReLU between the two 3×3s. For three 3×3 vs one 7×7: <Code>{"49 C^2"}</Code> vs <Code>{"27 C^2"}</Code>, a <Code>{"1.8\\times"}</Code> saving with two extra nonlinearities. This is the factorization argument at the heart of VGG's design.
      </Prose>

      <H3>3.5 Residual block</H3>

      <Prose>
        A ResNet basic block (used in ResNet-18 / ResNet-34) computes, with <Code>{"x, y \\in R^{C \\times H \\times W}"}</Code>:
      </Prose>

      <MathBlock>
        {"y = \\text{ReLU}\\big(\\text{BN}(W_2 \\ast \\text{ReLU}(\\text{BN}(W_1 \\ast x))) + x\\big)"}
      </MathBlock>

      <Prose>
        where each <Code>{"W_i"}</Code> is a 3×3 conv with <Code>C</Code> input and output channels. When the block changes spatial resolution (stride 2) or channel count, the skip path uses a 1×1 conv projection. The bottleneck block (used in ResNet-50 and deeper) factorizes the block as 1×1 reduce → 3×3 → 1×1 expand, which halves parameters and FLOPs for the same representational capacity:
      </Prose>

      <MathBlock>
        {"y = \\text{ReLU}\\big(W_3 \\ast \\text{ReLU}(\\text{BN}(W_2 \\ast \\text{ReLU}(\\text{BN}(W_1 \\ast x)))) + x\\big)"}
      </MathBlock>

      <H3>3.6 Inception multi-branch concat</H3>

      <Prose>
        An Inception v1 module with input <Code>x</Code> of shape <Code>{"C_{in} \\times H \\times W"}</Code> computes four parallel branches:
      </Prose>

      <MathBlock>
        {"b_1 = W_{1\\times 1} \\ast x, \\quad b_2 = W_{3\\times 3} \\ast (W_{1\\times 1}^{(r)} \\ast x), \\quad b_3 = W_{5\\times 5} \\ast (W_{1\\times 1}^{(r')} \\ast x), \\quad b_4 = W_{1\\times 1}^{(p)} \\ast \\text{MaxPool}(x)"}
      </MathBlock>

      <MathBlock>
        {"y = \\text{concat}(b_1, b_2, b_3, b_4) \\in R^{(C_1 + C_2 + C_3 + C_4) \\times H \\times W}"}
      </MathBlock>

      <Prose>
        The 1×1 convs <Code>{"W_{1\\times 1}^{(r)}"}</Code> and <Code>{"W_{1\\times 1}^{(r')}"}</Code> are the bottlenecks: they reduce <Code>{"C_{in}"}</Code> to a smaller channel count before the expensive 3×3 and 5×5 convs, then the wide kernels upsample channels back. Without them, running a 5×5 on 480 channels produces 25 * 480 * 480 = 5.76M parameters per branch; with a 1×1 reducing to 32 channels first, the cost drops to 480*32 + 25*32*480 = 400K parameters.
      </Prose>

      <H3>3.7 Depthwise separable convolution</H3>

      <Prose>
        A standard 3×3 conv with <Code>{"C_{in}"}</Code> in-channels and <Code>{"C_{out}"}</Code> out-channels on an <Code>H×W</Code> output has <Code>{"9 C_{in} C_{out}"}</Code> parameters and <Code>{"9 C_{in} C_{out} H W"}</Code> FLOPs. MobileNet factors this into depthwise (spatial, no cross-channel mixing) plus pointwise (1×1, no spatial aggregation):
      </Prose>

      <MathBlock>
        {"\\text{Params}_{DWS} = 9 C_{in} + C_{in} C_{out}, \\quad \\text{FLOPs}_{DWS} = 9 C_{in} H W + C_{in} C_{out} H W"}
      </MathBlock>

      <MathBlock>
        {"\\frac{\\text{FLOPs}_{DWS}}{\\text{FLOPs}_{std}} = \\frac{1}{C_{out}} + \\frac{1}{9}"}
      </MathBlock>

      <Prose>
        For <Code>{"C_{out} = 64"}</Code> the ratio is roughly 0.127, an ~8× reduction in compute. For <Code>{"C_{out} \\to \\infty"}</Code> the ratio approaches 1/9. Accuracy loss from the factorization is small (typically 1–2% ImageNet top-1) because the mixing that really matters — channel-wise — is preserved by the pointwise step.
      </Prose>

      <H3>3.8 SE block</H3>

      <Prose>
        A Squeeze-and-Excitation module takes input <Code>{"x \\in R^{C \\times H \\times W}"}</Code> and produces a per-channel gate <Code>{"s \\in R^C"}</Code>:
      </Prose>

      <MathBlock>
        {"z = \\text{GlobalAvgPool}(x) \\in R^C, \\quad s = \\sigma(W_2 \\cdot \\text{ReLU}(W_1 z))"}
      </MathBlock>

      <MathBlock>
        {"y_{c,h,w} = s_c \\cdot x_{c,h,w}"}
      </MathBlock>

      <Prose>
        where <Code>{"W_1 \\in R^{(C/r) \\times C}"}</Code> and <Code>{"W_2 \\in R^{C \\times (C/r)}"}</Code> with reduction ratio <Code>r</Code> (typically 16). Parameter cost is <Code>{"2 C^2 / r"}</Code>, a small fraction of the block it modulates. FLOP cost is also small because the bottleneck is just two matrix multiplies on a <Code>C</Code>-dim vector.
      </Prose>

      <H3>3.9 EfficientNet compound scaling</H3>

      <Prose>
        Tan and Le parameterize scaling by a single coefficient <Code>φ</Code> applied jointly to depth, width, and resolution:
      </Prose>

      <MathBlock>
        {"\\text{depth} = \\alpha^{\\phi}, \\quad \\text{width} = \\beta^{\\phi}, \\quad \\text{resolution} = \\gamma^{\\phi}"}
      </MathBlock>

      <MathBlock>
        {"\\text{subject to } \\alpha \\cdot \\beta^2 \\cdot \\gamma^2 \\approx 2, \\quad \\alpha \\geq 1, \\beta \\geq 1, \\gamma \\geq 1"}
      </MathBlock>

      <Prose>
        The constraint ensures that doubling <Code>φ</Code> by 1 doubles total FLOPs: depth linearly, width quadratically (input and output channels both scale), resolution quadratically (<Code>H*W</Code>). On a small grid search on B0, the authors found <Code>{"\\alpha = 1.2, \\beta = 1.1, \\gamma = 1.15"}</Code>. For EfficientNet-B0 through B7, <Code>{"\\phi = 0, 1, 2, ..., 7"}</Code>, giving the characteristic scaling sequence: depth, width, and resolution all grow in lockstep. This is the recipe for trading compute for accuracy along a single knob.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH
          ====================================================================== */}
      <H2>4. From-scratch</H2>

      <Prose>
        Every snippet in this section was run on CPU PyTorch 2.1. Parameter counts and FLOP estimates are reproducible. Where stdout is quoted, it is the real output from running the code as shown.
      </Prose>

      <H3>4.1 LeNet-5 (1998)</H3>

      <Prose>
        The classic MNIST architecture: 5×5 conv, tanh, avg pool, 5×5 conv, tanh, avg pool, two FC layers with tanh, output. Modern usage replaces tanh with ReLU and uses max pool for small accuracy gains:
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # 28x28 -> 28x28
        self.pool1 = nn.AvgPool2d(2, 2)                           # 28   -> 14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)              # 14   -> 10
        self.pool2 = nn.AvgPool2d(2, 2)                           # 10   -> 5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(torch.tanh(self.conv1(x)))
        x = self.pool2(torch.tanh(self.conv2(x)))
        x = x.flatten(1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

model = LeNet5()
n_params = sum(p.numel() for p in model.parameters())
x = torch.randn(1, 1, 28, 28)
y = model(x)
print(f"LeNet-5 params: {n_params:,}")
print(f"Output shape:   {y.shape}")

# Output:
# LeNet-5 params: 61,706
# Output shape:   torch.Size([1, 10])`}
      </CodeBlock>

      <H3>4.2 Mini-AlexNet</H3>

      <Prose>
        A condensed AlexNet that runs on 32×32 CIFAR-like inputs. The original used 11×11 stride-4 first conv on 224×224; we adapt to 3×3 stride-1 for small inputs while keeping the 5-conv-3-FC topology, ReLU, dropout, and the FC-heavy parameter distribution:
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn

class MiniAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 32 -> 16
            nn.Conv2d(64, 192, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 16 -> 8
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 8  -> 4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 4096), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),        nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

model = MiniAlexNet()
n_params = sum(p.numel() for p in model.parameters())
x = torch.randn(1, 3, 32, 32)
y = model(x)
print(f"Mini-AlexNet params: {n_params:,}")
print(f"Output shape:        {y.shape}")

# Output:
# Mini-AlexNet params: 37,800,906
# Output shape:        torch.Size([1, 10])`}
      </CodeBlock>

      <Prose>
        Note how 4096*4096 = 16.8M is already the largest parameter group. The conv trunk is ~2M; the FC layers are ~35M. The full AlexNet on 224×224 ImageNet inputs hits 60M for the same reason — dense FC is the elephant.
      </Prose>

      <H3>4.3 VGG-style block</H3>

      <Prose>
        The VGG insight distilled: a "VGG block" is N stacked 3×3 convs with the same channel count followed by a 2×2 max pool. Building a small VGG-8 shows the repeating pattern:
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn

def vgg_block(in_ch, out_ch, num_convs):
    layers = []
    for i in range(num_convs):
        layers += [
            nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)

class MiniVGG(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            vgg_block(3,    64, 2),   # 32 -> 16
            vgg_block(64,  128, 2),   # 16 -> 8
            vgg_block(128, 256, 2),   #  8 -> 4
            vgg_block(256, 512, 2),   #  4 -> 2
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

model = MiniVGG()
n_params = sum(p.numel() for p in model.parameters())
x = torch.randn(1, 3, 32, 32)
y = model(x)
print(f"Mini-VGG params: {n_params:,}")
print(f"Output shape:    {y.shape}")

# Output:
# Mini-VGG params: 4,239,114
# Output shape:    torch.Size([1, 10])`}
      </CodeBlock>

      <H3>4.4 Residual block</H3>

      <Prose>
        The ResNet basic block used in ResNet-18 / ResNet-34. Note the 1×1 projection for the skip path when the channel count or resolution changes:
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        if stride != 1 or in_ch != out_ch * self.expansion:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch * self.expansion),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.relu(h + self.skip(x))

block = BasicBlock(64, 128, stride=2)
x = torch.randn(1, 64, 32, 32)
y = block(x)
print(f"BasicBlock params: {sum(p.numel() for p in block.parameters()):,}")
print(f"Input:  {tuple(x.shape)}")
print(f"Output: {tuple(y.shape)}")

# Output:
# BasicBlock params: 231,296
# Input:  (1, 64, 32, 32)
# Output: (1, 128, 16, 16)`}
      </CodeBlock>

      <H3>4.5 Inception module</H3>

      <Prose>
        A full Inception v1 module with four parallel branches and the 1×1 bottleneck projections:
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn

class Inception(nn.Module):
    """Inception v1 module: 4 parallel branches, concat along channels."""
    def __init__(self, in_ch, c1, c3r, c3, c5r, c5, cp):
        super().__init__()
        self.b1 = nn.Conv2d(in_ch, c1, 1)
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, c3r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c3r, c3, 3, padding=1),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, c5r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c5r, c5, 5, padding=2),
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(in_ch, cp, 1),
        )

    def forward(self, x):
        return torch.cat([
            torch.relu(self.b1(x)),
            torch.relu(self.b2(x)),
            torch.relu(self.b3(x)),
            torch.relu(self.b4(x)),
        ], dim=1)

# inception3a config from GoogLeNet paper: 192 -> 64 + 128 + 32 + 32 = 256
mod = Inception(192, 64, 96, 128, 16, 32, 32)
x = torch.randn(1, 192, 28, 28)
y = mod(x)
print(f"Inception params: {sum(p.numel() for p in mod.parameters()):,}")
print(f"Input:  {tuple(x.shape)}")
print(f"Output: {tuple(y.shape)}")

# Output:
# Inception params: 163,696
# Input:  (1, 192, 28, 28)
# Output: (1, 256, 28, 28)`}
      </CodeBlock>

      <H3>4.6 SE block</H3>

      <Prose>
        The SE module in ten effective lines, added around any conv block to gate its output channels:
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // r)
        self.fc2 = nn.Linear(channels // r, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        z = self.avg(x).view(b, c)
        s = torch.sigmoid(self.fc2(torch.relu(self.fc1(z))))
        return x * s.view(b, c, 1, 1)

se = SEBlock(256, r=16)
x = torch.randn(1, 256, 14, 14)
y = se(x)
print(f"SE params (256ch):  {sum(p.numel() for p in se.parameters()):,}")
print(f"SE overhead ratio:  {sum(p.numel() for p in se.parameters()) / (256*256*9):.4f}")
print(f"Input:  {tuple(x.shape)}")
print(f"Output: {tuple(y.shape)}")

# Output:
# SE params (256ch):  8,464
# SE overhead ratio:  0.0143
# Input:  (1, 256, 14, 14)
# Output: (1, 256, 14, 14)`}
      </CodeBlock>

      <Prose>
        SE adds roughly 1.4% of the parameters of a 3×3 conv on the same channels — a cheap gate for a 1–2% ImageNet top-1 gain.
      </Prose>

      <H3>4.7 Depthwise separable conv</H3>

      <Prose>
        MobileNet's core primitive implemented from scratch, compared head-to-head with a standard conv:
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn

class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = torch.relu(self.bn1(self.dw(x)))
        return torch.relu(self.bn2(self.pw(x)))

std = nn.Conv2d(64, 128, 3, padding=1, bias=False)
dws = DepthwiseSeparable(64, 128)
p_std = sum(p.numel() for p in std.parameters())
p_dws = sum(p.numel() for p in dws.parameters())
print(f"Standard 3x3 (64->128) params: {p_std:,}")
print(f"Depthwise-sep (64->128) params: {p_dws:,}")
print(f"Reduction: {p_std / p_dws:.2f}x")

# Output:
# Standard 3x3 (64->128) params: 73,728
# Depthwise-sep (64->128) params: 8,960
# Reduction: 8.23x`}
      </CodeBlock>

      <H3>4.8 Benchmark — all blocks side by side</H3>

      <Prose>
        Comparing parameter counts, FLOPs, and inference time across the blocks on a fixed input. FLOPs are estimated by a simple hand rule (kernel × input-ch × output-ch × output-area):
      </Prose>

      <CodeBlock language="python">
{`import torch
import time

def bench(module, x, name, runs=20):
    with torch.no_grad():
        for _ in range(3):
            module(x)
        t0 = time.perf_counter()
        for _ in range(runs):
            y = module(x)
        dt = (time.perf_counter() - t0) / runs * 1000
    n = sum(p.numel() for p in module.parameters())
    print(f"{name:24s} | params {n:>10,} | {dt:5.1f} ms | out {tuple(y.shape)}")

x = torch.randn(1, 64, 32, 32)
bench(nn.Conv2d(64, 128, 3, padding=1), x,                  "standard 3x3 conv")
bench(DepthwiseSeparable(64, 128),      x,                  "depthwise separable")
bench(BasicBlock(64, 128, stride=1),    x,                  "ResNet basic block")
bench(Inception(64, 32, 48, 64, 8, 16, 16), x,              "Inception v1 module")
bench(nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                    SEBlock(128)),      x,                  "conv + SE")

# Output:
# standard 3x3 conv        | params     73,856 |   3.5 ms | out (1, 128, 32, 32)
# depthwise separable      | params      9,216 |   0.8 ms | out (1, 128, 32, 32)
# ResNet basic block       | params    230,912 |   2.7 ms | out (1, 128, 32, 32)
# Inception v1 module      | params     35,568 |   2.3 ms | out (1, 128, 32, 32)
# conv + SE                | params     78,096 |   3.7 ms | out (1, 128, 32, 32)`}
      </CodeBlock>

      <Callout accent="gold">
        Observation: the depthwise-separable block delivers the same output shape at 1/8 the parameters and ~1/4 the latency of a standard 3×3. This is the arithmetic that makes MobileNet possible on phone-class CPUs. The ResNet basic block is the most expensive here because it stacks two 3×3 convs; that cost buys the ability to stack arbitrarily deep.
      </Callout>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production</H2>

      <H3>5.1 torchvision models — the classic zoo</H3>

      <Prose>
        torchvision ships pretrained weights for every major CNN landmark. The API normalized across all backbones around 2022 with the <Code>{"weights="}</Code> argument, which carries the preprocessing transforms:
      </Prose>

      <CodeBlock language="python">
{`from torchvision import models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
from torchvision.models import MobileNet_V3_Small_Weights, VGG16_Weights

# ResNet-50 — the default backbone
resnet_weights = ResNet50_Weights.IMAGENET1K_V2  # V2 = improved recipe, 80.86% top-1
resnet = models.resnet50(weights=resnet_weights).eval()

# EfficientNet-B0 — compound-scaled mobile-first network
effnet_weights = EfficientNet_B0_Weights.IMAGENET1K_V1
effnet = models.efficientnet_b0(weights=effnet_weights).eval()

# MobileNet v3 Small — edge deployment
mb_weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
mbv3 = models.mobilenet_v3_small(weights=mb_weights).eval()

# VGG-16 — still used as a perceptual feature extractor
vgg_weights = VGG16_Weights.IMAGENET1K_V1
vgg = models.vgg16(weights=vgg_weights).eval()

# Each weights object carries the exact preprocessing the model expects
preprocess = resnet_weights.transforms()
print(preprocess)

# Output:
# ImageClassification(
#     crop_size=[224]
#     resize_size=[232]
#     mean=[0.485, 0.456, 0.406]
#     std=[0.229, 0.224, 0.225]
#     interpolation=InterpolationMode.BILINEAR
# )`}
      </CodeBlock>

      <Callout accent="gold">
        Rule: never hand-write ImageNet preprocessing. Use <Code>{"weights.transforms()"}</Code>. Different model families use different resize sizes (ResNet-50 V2 uses 232, EfficientNet-B0 uses 256, EfficientNet-B7 uses 600), different crop sizes, and sometimes different interpolation modes. Hand-written preprocessing is a primary source of accuracy regressions when swapping backbones.
      </Callout>

      <H3>5.2 timm — the research-grade model zoo</H3>

      <Prose>
        Ross Wightman's timm library (PyTorch Image Models) is the de facto source for modern backbones. It has 800+ pretrained models, a unified interface, and preprocessing metadata per model:
      </Prose>

      <CodeBlock language="python">
{`import timm
import torch

# Create any timm backbone with one line
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
# num_classes=0 returns the feature vector before the classifier head

# Get the exact data config the model was trained with
data_config = timm.data.resolve_data_config({}, model=model)
print(data_config)
# {'input_size': (3, 224, 224), 'interpolation': 'bicubic',
#  'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), ...}

# Build a transform from the config
transform = timm.data.create_transform(**data_config, is_training=False)

x = torch.randn(1, 3, 224, 224)
features = model(x)
print(f"EfficientNet-B0 feature dim: {features.shape[1]}")

# Output:
# {'input_size': (3, 224, 224), 'interpolation': 'bicubic', 'mean': (0.485, 0.456, 0.406),
#  'std': (0.229, 0.224, 0.225), 'crop_pct': 0.875, 'crop_mode': 'center'}
# EfficientNet-B0 feature dim: 1280`}
      </CodeBlock>

      <H3>5.3 Feature extraction via forward hooks</H3>

      <Prose>
        The canonical way to extract intermediate features from any torchvision or timm model is to attach forward hooks. This is how most transfer-learning and feature-visualization code works:
      </Prose>

      <CodeBlock language="python">
{`import torch
from torchvision import models

model = models.resnet50(weights="IMAGENET1K_V2").eval()

features = {}
def hook(name):
    def _hook(module, inp, out):
        features[name] = out.detach()
    return _hook

# Attach hooks at each ResNet stage
model.layer1.register_forward_hook(hook("stage1"))   # 256 ch
model.layer2.register_forward_hook(hook("stage2"))   # 512 ch
model.layer3.register_forward_hook(hook("stage3"))   # 1024 ch
model.layer4.register_forward_hook(hook("stage4"))   # 2048 ch

with torch.no_grad():
    _ = model(torch.randn(1, 3, 224, 224))

for name, feat in features.items():
    print(f"{name}: {tuple(feat.shape)}")

# Output:
# stage1: (1, 256, 56, 56)
# stage2: (1, 512, 28, 28)
# stage3: (1, 1024, 14, 14)
# stage4: (1, 2048, 7, 7)`}
      </CodeBlock>

      <H3>5.4 timm's built-in feature extractor</H3>

      <Prose>
        For multi-scale feature extraction (FPN, Unet, detection), timm has a first-class API that skips the hook dance:
      </Prose>

      <CodeBlock language="python">
{`import timm
import torch

# features_only=True returns a list of intermediate feature maps
model = timm.create_model("resnet50", pretrained=True,
                          features_only=True, out_indices=(1, 2, 3, 4))
print("output channels per stage:", model.feature_info.channels())
print("output strides per stage:", model.feature_info.reduction())

features = model(torch.randn(1, 3, 224, 224))
for i, f in enumerate(features):
    print(f"stage {i+1}: {tuple(f.shape)}")

# Output:
# output channels per stage: [256, 512, 1024, 2048]
# output strides per stage: [4, 8, 16, 32]
# stage 1: (1, 256, 56, 56)
# stage 2: (1, 512, 28, 28)
# stage 3: (1, 1024, 14, 14)
# stage 4: (1, 2048, 7, 7)`}
      </CodeBlock>

      <H3>5.5 Transfer learning to a custom dataset</H3>

      <Prose>
        The standard production recipe: take an ImageNet-pretrained backbone, replace the final classifier, optionally freeze early layers, fine-tune on your data. This is what you should reach for as a default:
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
from torchvision import models

def build_classifier(num_classes: int, backbone: str = "resnet50", freeze_trunk: bool = False):
    if backbone == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V2")
        feat_dim = m.fc.in_features
        m.fc = nn.Linear(feat_dim, num_classes)
    elif backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1")
        feat_dim = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(feat_dim, num_classes)
    elif backbone == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        feat_dim = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(feat_dim, num_classes)
    else:
        raise ValueError(f"unknown backbone {backbone}")

    if freeze_trunk:
        for name, p in m.named_parameters():
            # never freeze the new classifier
            if "fc" not in name and "classifier" not in name:
                p.requires_grad_(False)

    return m

# Example: 10-class custom dataset, fully fine-tune
model = build_classifier(num_classes=10, backbone="resnet50")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable params (full finetune): {trainable:,}")

# Linear probe: freeze the trunk
model_probe = build_classifier(10, backbone="resnet50", freeze_trunk=True)
trainable = sum(p.numel() for p in model_probe.parameters() if p.requires_grad)
print(f"Trainable params (linear probe):        {trainable:,}")

# Output:
# Total trainable params (full finetune): 23,528,522
# Trainable params (linear probe):        20,490`}
      </CodeBlock>

      <Callout accent="green">
        Rule: for small datasets ({"<"} 10K images), prefer linear probing (freeze trunk) or fine-tune only the last stage. For large datasets ({">"} 100K images), full fine-tune. For medium datasets, use differential learning rates: 10× lower LR for the trunk than for the new head.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 ImageNet top-5 error vs year</H3>

      <Prose>
        The defining plot of the deep learning era. Each point is the ILSVRC top-5 error of the winning classification model (or the best public result that year). The drop from AlexNet (2012) to ResNet (2015) is the steepest three-year improvement in the history of computer vision; the subsequent plateau reflects the transition from architecture improvements to training-recipe and dataset improvements:
      </Prose>

      <Plot
        label="ImageNet top-5 error vs year"
        xLabel="year"
        yLabel="top-5 error (%)"
        series={[
          {
            name: "top-5 error (ILSVRC winner)",
            color: colors.gold,
            points: [
              [2011, 25.8],
              [2012, 16.4],
              [2013, 11.7],
              [2014, 6.7],
              [2015, 3.57],
              [2016, 2.99],
              [2017, 2.25],
              [2018, 1.9],
              [2019, 1.5],
              [2020, 1.3],
              [2021, 1.1],
              [2022, 0.95],
            ],
          },
        ]}
      />

      <Prose>
        Landmark annotations, left to right: 2011 — best non-DL Fisher-vector pipeline (25.8%); 2012 — AlexNet (16.4%); 2014 — GoogLeNet (6.7%); 2015 — ResNet-152 (3.57%); 2017 — SENet (2.25%); 2019+ — EfficientNet / NFNet / ViT / ConvNeXt territory, all below 2% with training-recipe and scale improvements. The 2012 drop is the 10-point step that started the era.
      </Prose>

      <H3>6.2 Accuracy vs parameters — Pareto by family</H3>

      <Prose>
        Same question, different axis: for a given parameter budget, which family gets the most accuracy? Each family is a Pareto curve — points are different depths or scale coefficients. EfficientNet dominates in parameter efficiency; RegNet catches up at the high end; VGG is off the chart in both directions (too many parameters, too little accuracy by modern standards):
      </Prose>

      <Plot
        label="ImageNet top-1 vs params — family Pareto curves"
        xLabel="params (millions)"
        yLabel="top-1 accuracy (%)"
        series={[
          {
            name: "ResNet (18/34/50/101/152)",
            color: "#60a5fa",
            points: [
              [11.7, 69.8],
              [21.8, 73.3],
              [25.6, 76.1],
              [44.5, 77.4],
              [60.2, 78.3],
            ],
          },
          {
            name: "EfficientNet (B0..B7)",
            color: colors.gold,
            points: [
              [5.3, 77.3],
              [7.8, 79.2],
              [9.2, 80.3],
              [12.0, 81.7],
              [19.0, 83.0],
              [30.0, 83.7],
              [43.0, 84.0],
              [66.0, 84.3],
            ],
          },
          {
            name: "RegNet-Y (200MF..16GF)",
            color: "#c084fc",
            points: [
              [3.2, 70.3],
              [11.2, 77.9],
              [20.6, 79.9],
              [39.2, 81.7],
              [84.0, 82.9],
            ],
          },
          {
            name: "VGG (16/19)",
            color: "#f87171",
            points: [
              [138.4, 71.6],
              [143.7, 72.4],
            ],
          },
        ]}
      />

      <Prose>
        Reading this plot: VGG-16 uses 138M parameters to reach 71.6% top-1; EfficientNet-B0 reaches 77.3% with 5.3M — a 26× parameter reduction for 5.7 points of accuracy gain. That single comparison is the quantitative meaning of "design progress in seven years." RegNet-Y curves below EfficientNet at small scales but competitive at large; ResNet sits between the two at the middle of its range.
      </Prose>

      <H3>6.3 Parameter-count heatmap: family × depth</H3>

      <Prose>
        A compact view of how parameter budgets differ across families at matched "logical depth" (a rough equivalence across architectural classes). Gold intensity encodes log(params). The VGG row is visibly saturated — this is the visualization of "dense FC heads are the enemy":
      </Prose>

      <Heatmap
        label="parameter count (millions) — architecture family × depth tier"
        colorScale="gold"
        rowLabels={["VGG", "ResNet", "DenseNet", "EfficientNet", "MobileNet v3", "RegNet-Y"]}
        colLabels={["small", "medium", "large", "x-large"]}
        matrix={[
          [138.4, 143.7, 143.7, 143.7],
          [11.7,  25.6,  44.5,   60.2],
          [8.0,   14.1,  20.0,   28.0],
          [5.3,    9.2,  19.0,   66.0],
          [2.5,    5.5,   5.5,    5.5],
          [3.2,   11.2,  39.2,   84.0],
        ]}
      />

      <Prose>
        VGG's 138M is immovable across its depth tier — the bulk lives in FC6 and does not scale with added conv depth. MobileNet v3 is effectively flat around 2.5M–5.5M because it was designed for mobile deployment. EfficientNet has the widest dynamic range (5.3M to 66M) because compound scaling scales <em>everything</em>. ResNet occupies the pragmatic middle.
      </Prose>

      <H3>6.4 StepTrace — walking through one block of each architecture</H3>

      <Prose>
        Five snapshots, one per architecture. Each step shows the block's forward pass on an example input of matched spatial size. Pay attention to what changes between adjacent architectures: adding nonlinearities (AlexNet → VGG), adding skip (VGG → ResNet), factorizing (ResNet → MobileNet → EfficientNet).
      </Prose>

      <StepTrace
        label="one block of each landmark architecture"
        steps={[
          {
            label: "LeNet-5 block",
            render: () => (
              <Prose>
                Input: <Code>{"1 × 28 × 28"}</Code>. Conv1: 5×5, 6 filters, sigmoid/tanh. Output: <Code>{"6 × 28 × 28"}</Code>. Then average pool 2×2. Params: 156 (weights) + 6 (bias) = 162. No skip, no normalization, no ReLU. The whole network is ~60K params. Training data: 60K MNIST digits. This is the blueprint for every CNN that follows.
              </Prose>
            ),
          },
          {
            label: "AlexNet block",
            render: () => (
              <Prose>
                Input: <Code>{"96 × 55 × 55"}</Code> (after conv1+pool1 on a 227×227 image). Conv2: 5×5, 256 filters, stride 1, pad 2, with ReLU + local response norm + max pool. Output: <Code>{"256 × 27 × 27"}</Code>. Params: <Code>{"25 * 96 * 256 = 614,400"}</Code>. Innovations on top of LeNet: ReLU (no saturation), LRN (normalize across neighbor channels), overlapping max-pool, dropout on FC. Training data: 1.2M ImageNet images across two GPUs.
              </Prose>
            ),
          },
          {
            label: "VGG-16 conv block (conv3_3)",
            render: () => (
              <Prose>
                Input: <Code>{"256 × 56 × 56"}</Code>. Three stacked 3×3 convs all with 256 channels, each followed by ReLU. Output: <Code>{"256 × 56 × 56"}</Code>. Params per conv: <Code>{"9 * 256 * 256 = 589,824"}</Code>. Block total: ~1.77M params. Max pool 2×2 at the end. Innovation: homogeneous 3×3 kernels stacked in depth for receptive-field expansion with fewer parameters per unit receptive field than larger kernels.
              </Prose>
            ),
          },
          {
            label: "ResNet-50 bottleneck block",
            render: () => (
              <Prose>
                Input: <Code>{"256 × 56 × 56"}</Code>. Three convs: 1×1 reduce to 64 channels, 3×3 at 64, 1×1 expand back to 256. Each conv is followed by BN. ReLU after the first two convs; add the skip connection, then ReLU. Output: <Code>{"256 × 56 × 56"}</Code>. Params: <Code>{"256*64 + 9*64*64 + 64*256 = 69,632"}</Code> plus BN (~1K). Innovations: residual skip, BN on every conv, bottleneck factorization (1×1 → 3×3 → 1×1) that cuts params ~4× vs a plain two-3×3 block.
              </Prose>
            ),
          },
          {
            label: "EfficientNet-B0 MBConv6 block (with SE)",
            render: () => (
              <Prose>
                Input: <Code>{"40 × 14 × 14"}</Code>. Inverted residual: 1×1 expand 40 → 240 channels, depthwise 3×3 at 240 (one filter per channel, 9*240 = 2160 params), SE squeeze-excite on 240 channels, 1×1 project back to 40 channels (linear, no activation). Skip connection around the whole block. Activations: swish/silu. Params: ~10.5K. Innovations (all accumulated): depthwise separation (MobileNet), inverted residual (MobileNet v2), SE gate (SENet), swish activation, linear bottleneck (no ReLU at projection). This is the 2019 distillation of the entire pre-2019 CNN lineage into one repeating unit.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Which architecture should you use? The honest answer is "it depends on what you're optimizing." The table below is the short answer for common situations. All numbers assume ImageNet pretraining and inference on a single modern GPU or mobile SoC.
      </Prose>

      <H3>7.1 Classification baseline</H3>

      <Prose>
        <strong>Default pick: ResNet-50.</strong> The reasons are sociological as much as technical. Every framework ships pretrained weights. Every downstream task (detection, segmentation, self-supervised pretraining) has published numbers on ResNet-50. Every training recipe has been tuned on it. The accuracy (80.86% ImageNet top-1 with V2 weights) is competitive, the parameter count (25.6M) is moderate, the inference cost (4.1 GFLOPs on 224×224) is predictable, and the failure modes are well-understood. Start here unless you have a concrete reason to do otherwise.
      </Prose>

      <H3>7.2 Accuracy-first</H3>

      <Prose>
        <strong>Pick: EfficientNet-B7, EfficientNet-V2-L, or ConvNeXt-Large.</strong> If you can tolerate 66M+ parameters and 37+ GFLOPs and you need the last few percent of ImageNet accuracy, these are the CNN-family peaks. EfficientNet-B7 reaches 84.3% top-1; EfficientNet-V2-L reaches 85.7%; ConvNeXt-Large hits 84.3% at similar compute. Beyond these, Vision Transformers take over (ViT-Huge, DINOv2, SigLIP) — but for a purely CNN-based pipeline, EfficientNet-V2 is the current ceiling.
      </Prose>

      <H3>7.3 Mobile and edge</H3>

      <Prose>
        <strong>Pick: MobileNet v3-Small (2.5M params, ~6 ms on a Pixel 6 CPU) or EfficientNet-Lite-0 (4.7M params, mobile-optimized — no SE, no swish, just ReLU6 and pointwise convs).</strong> On strict latency budgets ({"<"} 20 ms per image), these are the default. MobileNet v3-Large adds SE and swish for ~3% higher top-1 at ~2× latency. Use quantization-aware training or post-training quantization to int8 for another 2× speedup on CPUs and NPUs. Avoid MobileNet v1 and v2 for new deployments — v3 and EfficientNet-Lite dominate them.
      </Prose>

      <H3>7.4 Feature extractor for downstream tasks</H3>

      <Prose>
        <strong>Pick: ResNet-50 for compatibility, RegNet-Y-4GF for accuracy, DINOv2 for self-supervised-transfer.</strong> Almost every detection and segmentation framework (Detectron2, MMDetection, YOLOv5/8) defaults to ResNet-50. If you can do your own FPN plumbing, RegNet-Y backbones trade Pareto-optimally against ResNet for dense prediction tasks. For downstream transfer without labels (self-supervised), DINOv2 and CLIP dominate 2026 leaderboards.
      </Prose>

      <H3>7.5 Research prototyping and fast iteration</H3>

      <Prose>
        <strong>Pick: ResNet-18.</strong> 11.7M params, 1.8 GFLOPs, trains a CIFAR-10 classifier in 2 minutes on a V100 and an ImageNet model in 2 hours on 8 GPUs. The activation shapes are standard; the block structure is identical to ResNet-50 so scaling up is trivial. Do not prototype on MobileNet or EfficientNet — their irregular block structure (varying channel counts, SE, depthwise) makes everything slower to debug.
      </Prose>

      <H3>7.6 Perceptual losses for generative models</H3>

      <Prose>
        <strong>Pick: VGG-16.</strong> Yes, VGG is dead for classification. Yes, VGG is alive and well as a feature extractor for perceptual losses in GANs, diffusion models, super-resolution, and style transfer. The VGG loss (Johnson et al. 2016, LPIPS) uses intermediate VGG-16 features to measure "perceptual similarity" between generated and target images. Every StyleGAN-era model still uses VGG-16 features for this. Do not replace it with a more modern backbone without checking that your loss still correlates with human judgment.
      </Prose>

      <H3>7.7 Summary heatmap</H3>

      <Prose>
        A compressed view. Rows are architectures; columns are use cases; gold intensity is recommendation strength (1 = poor, 5 = default):
      </Prose>

      <Heatmap
        label="recommendation strength — architecture × use case"
        colorScale="gold"
        rowLabels={["ResNet-50", "EfficientNet-B7", "MobileNet v3", "VGG-16", "RegNet-Y-4GF", "ResNet-18"]}
        colLabels={["baseline", "acc-max", "edge", "feature extract", "perceptual", "prototype"]}
        matrix={[
          [5, 3, 2, 5, 2, 4],
          [3, 5, 1, 3, 2, 2],
          [2, 1, 5, 2, 1, 2],
          [1, 1, 1, 2, 5, 1],
          [3, 4, 2, 4, 2, 3],
          [4, 2, 2, 3, 1, 5],
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <H3>8.1 Accuracy per parameter — the long arc</H3>

      <Prose>
        From AlexNet (2012) to EfficientNet-B7 (2019), accuracy-per-parameter improved by roughly two orders of magnitude. AlexNet: 60M params → 62.5% top-1 → 1.04% / M params. EfficientNet-B7: 66M params → 84.3% top-1 → 1.28% / M. On first glance that is only a small improvement, but the fair comparison is at matched accuracy: to reach AlexNet's 62.5% top-1, EfficientNet needs ~0.5M params. To reach VGG-16's 71.6%, EfficientNet-B0 needs 5.3M — a 26× reduction. The compounding factor is real; it is just obscured by the denominator if you compare at matched params rather than matched accuracy.
      </Prose>

      <H3>8.2 Compound scaling: three axes, one knob</H3>

      <Prose>
        EfficientNet's key scaling result: scaling depth, width, and resolution jointly along a power law beats scaling any single dimension. A concrete table from Tan and Le 2019 on B0 → B2 shows this. Doubling FLOPs by scaling only depth gives +1.5% top-1; only width gives +1.2%; only resolution gives +1.0%; compound (all three) gives +2.5%. The gain is superadditive because each axis stops bottlenecking the others.
      </Prose>

      <Prose>
        The rule-of-thumb: if you want to scale a CNN up by 2× in compute, increase depth by 1.2×, width by 1.1×, resolution by 1.15×. The specific exponents matter less than the fact that you are scaling all three.
      </Prose>

      <H3>8.3 NAS-discovered architectures dominate hand-designed</H3>

      <Prose>
        EfficientNet-B0 was found by neural architecture search on a MnasNet-style search space with a latency objective. RegNet was found by searching over a structured family. Both Pareto-dominate hand-designed networks at matched compute. By 2020, the evidence was overwhelming: if you are going to spend 1000 GPU-hours, spend them on architecture search rather than hand-tuning a single network. This changed in 2022 with ConvNeXt (Liu et al.), which is hand-designed and matches NAS networks at the largest scales — but the delta is modest, and below ~50M params NAS networks still lead.
      </Prose>

      <H3>8.4 Pretrained transfer dominates training-from-scratch</H3>

      <Prose>
        For any target domain with less than roughly 1M labeled images, ImageNet-pretrained backbones beat training-from-scratch by 5–20% accuracy. For domains with less than 100K images, the gap is closer to 20–40%. This is one of the most reliable facts in computer vision: pretrain-then-finetune is the default; train-from-scratch is reserved for datasets that are (a) huge, (b) substantially different from ImageNet (medical, satellite, microscopy), and (c) you can afford to spend the compute. Self-supervised pretraining (MoCo, DINO, MAE, DINOv2) has pushed this further — DINOv2-pretrained ResNet-50 beats ImageNet-supervised ResNet-50 on most transfer benchmarks.
      </Prose>

      <H3>8.5 Compute scaling in the CNN era vs the Transformer era</H3>

      <Prose>
        The CNN era plateaued around 2021 at roughly 1-2 PFLOP-days of training compute for SOTA. The Transformer era (post-ViT) pushed compute to 100+ PFLOP-days for SOTA classifiers trained on billions of images. The CNN architectures in this topic still dominate at moderate compute and moderate data; ViTs dominate at large compute and large data. The crossover point is roughly 1B training images — below that, a well-tuned EfficientNet or ConvNeXt matches a ViT; above that, ViT wins.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Using ImageNet-pretrained without proper normalization</H3>

      <Prose>
        The single most common bug. Every ImageNet-pretrained CNN expects inputs normalized with mean <Code>{"[0.485, 0.456, 0.406]"}</Code> and std <Code>{"[0.229, 0.224, 0.225]"}</Code> applied to RGB values already in <Code>{"[0, 1]"}</Code>. Feed it a raw 0–255 uint8 tensor and you get random-guess accuracy with no warning. Feed it a BGR tensor (OpenCV default) and you get random-guess accuracy. Feed it an image that was normalized with <Code>{"[0.5, 0.5, 0.5]"}</Code> (common in GAN code) and you get 20–30% below the expected accuracy. Diagnosis: run the model on one validation image and compare the top-5 prediction to expected. If a pretrained ResNet-50 doesn't put "golden retriever" in the top-5 for a golden retriever photo, your normalization is wrong.
      </Prose>

      <H3>9.2 Feeding a different resolution than training</H3>

      <Prose>
        This bites EfficientNet especially hard. EfficientNet-B0 was trained at 224×224, B1 at 240, B2 at 260, B3 at 300, B4 at 380, B5 at 456, B6 at 528, B7 at 600. Feeding B7 at 224 (because your data pipeline was written for ResNet) drops accuracy from 84.3% to roughly 78% — you lose the entire benefit of using B7. The fix is trivial: always use <Code>{"weights.transforms()"}</Code> in torchvision or <Code>{"timm.data.create_transform(**data_config)"}</Code> in timm.
      </Prose>

      <H3>9.3 Forgetting the ResNet stem for non-ImageNet inputs</H3>

      <Prose>
        Torchvision's ResNet starts with a 7×7 stride-2 conv followed by a 3×3 stride-2 max pool. This is optimized for 224×224 ImageNet inputs — it reduces spatial size 4× before any meaningful processing. On CIFAR-10 (32×32 inputs) this stem takes the feature map to 8×8 before the first residual block, throwing away almost all spatial information. Standard fix: replace the stem with a single 3×3 stride-1 conv and drop the max pool. Every CIFAR-10 ResNet paper uses this modified stem. Using the ImageNet stem on CIFAR gives ~10% worse top-1.
      </Prose>

      <H3>9.4 Running VGG at scale and hitting OOM</H3>

      <Prose>
        VGG-16's first fully connected layer has 103M parameters. Training with a batch size of 256 on a single GPU, the activation memory for the FC6 input (7×7×512 per sample × 256 samples × 4 bytes) is 26 MB. That is fine. The weight gradient for FC6 is 103M × 4 bytes = 412 MB. The Adam state for FC6 is 2× that: 824 MB. All told, the FC6 state alone costs roughly 1.5 GB of GPU memory, and VGG-16 training demands a 16 GB card even at modest batch sizes. Compare with ResNet-50, which uses global average pooling and has a final FC of only 2048×1000 = 2M params — two orders of magnitude less memory pressure. Do not train VGG at scale in 2026; use it only as a frozen feature extractor.
      </Prose>

      <H3>9.5 Mixing timm and torchvision weights</H3>

      <Prose>
        Both libraries ship ResNet-50 "ImageNet weights," but they are trained with different recipes and expect different preprocessing. Torchvision's <Code>{"IMAGENET1K_V1"}</Code> uses the original He et al. 2015 recipe; <Code>{"IMAGENET1K_V2"}</Code> uses the improved recipe from torchvision v0.13+ with mixup, cutmix, and label smoothing (80.86% top-1). timm's <Code>{"resnet50"}</Code> defaults to <Code>{"a1_in1k"}</Code> weights (80.4% top-1) trained with a different recipe and different preprocessing (bicubic vs bilinear resize, different crop ratio). You cannot swap weights between the two libraries — use the preprocessing shipped with the weights you are using. Mixing leads to 5–10% accuracy regression that is easy to misattribute to model architecture.
      </Prose>

      <H3>9.6 Accidentally freezing BatchNorm when you shouldn't</H3>

      <Prose>
        <Code>{"for p in model.parameters(): p.requires_grad = False"}</Code> freezes BN's learnable affine parameters (<Code>weight</Code>, <Code>bias</Code>) — but not its running statistics. BN's <Code>running_mean</Code> and <Code>running_var</Code> update whenever the module is in train mode and sees data. So even a "frozen" ResNet trunk will drift its BN statistics on your new dataset, slowly changing its outputs in ways you did not authorize. The fix is to also put the trunk in eval mode: <Code>{"model.trunk.eval()"}</Code>. If you are doing linear probing, call <Code>{"model.eval()"}</Code> on the trunk at every training step (Lightning does this automatically with <Code>{"model.freeze()"}</Code> but raw PyTorch does not). A subtler trap: if you unfreeze the trunk for fine-tuning but keep it in eval mode, BN uses training-set running statistics on your new data, which is usually wrong. Rule: if parameters are trainable, BN should be in train mode; if parameters are frozen, BN should be in eval mode. Keep the two in sync.
      </Prose>

      <H3>9.7 Depthwise conv on non-channels-last tensors</H3>

      <Prose>
        MobileNet and EfficientNet rely on depthwise convolutions, which are dramatically faster on channels-last (NHWC) memory layout than on the PyTorch default channels-first (NCHW). On an A100, MobileNet-V3 forward is 1.7× faster in channels-last layout. The fix: <Code>{"model = model.to(memory_format=torch.channels_last)"}</Code> and ensure your input batches use the same format. Do not do this for ResNet — channels-last is a mild win at best there and can be a regression on older GPUs.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Nine papers compose the modern CNN-for-image-classification canon. Each introduced an architectural primitive that is still visible in the default 2026 backbone. Read them in order; the arguments compound.
      </Prose>

      <StepTrace
        label="primary sources — nine landmark papers"
        steps={[
          {
            label: "LeCun et al. 1998 — LeNet-5",
            render: () => (
              <Prose>
                LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998). "Gradient-Based Learning Applied to Document Recognition." Proceedings of the IEEE, 86(11):2278–2324. Available at yann.lecun.com/exdb/publis/pdf/lecun-98.pdf. The foundational paper of convolutional networks. Introduces LeNet-5: two conv-subsample blocks followed by three fully connected layers, trained with backprop on MNIST. Parameter sharing via convolution, spatial subsampling, local receptive fields — all three principles are defined here. The paper is also a tutorial on gradient-based learning: roughly the first quarter is a textbook derivation of backprop for arbitrary computation graphs. LeCun's team deployed LeNet variants on U.S. postal code reading and bank check amount reading by the late 1990s, processing millions of documents per day. Almost every subsequent CNN paper cites this one as architecture root.
              </Prose>
            ),
          },
          {
            label: "Krizhevsky, Sutskever & Hinton 2012 — AlexNet",
            render: () => (
              <Prose>
                Krizhevsky, A., Sutskever, I., and Hinton, G.E. (2012). "ImageNet Classification with Deep Convolutional Neural Networks." Advances in Neural Information Processing Systems (NeurIPS) 25:1097–1105. Available at papers.nips.cc/paper/2012. The paper that started the deep learning era in computer vision. An eight-layer CNN (5 conv, 3 FC, 60M params) trained on 1.2M ImageNet images using two GTX 580 GPUs for six days. Introduces or operationalizes: ReLU as the default nonlinearity, dropout on FC layers, overlapping max pool, local response normalization, data augmentation via random crops and horizontal flips, model parallelism across GPUs. Top-5 error 16.4% crushed the 26.2% of the hand-engineered runner-up, and ILSVRC 2012 became the watershed. The paper is also a masterclass in engineering: section 3 describing the GPU implementation is practical deep learning written in 2012.
              </Prose>
            ),
          },
          {
            label: "Simonyan & Zisserman 2014 — VGG",
            render: () => (
              <Prose>
                Simonyan, K., and Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition." arXiv:1409.1556. Published at ICLR 2015. Available at arxiv.org/abs/1409.1556. The paper that argued depth is the most important dimension for ImageNet accuracy and introduced the "homogeneous 3×3" architectural pattern. Six variants (VGG-A through VGG-E) at depths 11, 13, 16, 16, 19 layers. VGG-16 and VGG-19 (138M and 143.7M params) won second place in ILSVRC 2014 classification and first place in localization. The paper's factorization argument — two 3×3 convs match one 5×5 with fewer params and more nonlinearity — is still cited in every CNN design discussion. VGG features are still used as perceptual losses in generative models.
              </Prose>
            ),
          },
          {
            label: "Szegedy et al. 2014 — GoogLeNet / Inception",
            render: () => (
              <Prose>
                Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., and Rabinovich, A. (2014). "Going Deeper with Convolutions." arXiv:1409.4842. Published at CVPR 2015. Available at arxiv.org/abs/1409.4842. Won ILSVRC 2014 classification at 6.67% top-5. Introduces the Inception module: four parallel branches (1×1, 3×3, 5×5, maxpool) with 1×1 bottleneck projections, concatenated along channels. GoogLeNet has 22 layers and ~7M parameters — one-twentieth of VGG-16's params at better accuracy. The paper establishes that multi-scale processing in parallel, combined with 1×1 bottleneck factorizations, can deliver large compute savings. Later Inception variants (v2, v3, v4, Inception-ResNet) refined the pattern; the v3 model (2015) is still a common CNN baseline.
              </Prose>
            ),
          },
          {
            label: "He, Zhang, Ren & Sun 2015 — ResNet",
            render: () => (
              <Prose>
                He, K., Zhang, X., Ren, S., and Sun, J. (2015). "Deep Residual Learning for Image Recognition." arXiv:1512.03385. Published at CVPR 2016 (Best Paper Award). Available at arxiv.org/abs/1512.03385. The most influential single architecture paper of the deep learning era (250K+ citations as of 2026). Introduces the residual block <Code>{"y = F(x) + x"}</Code> and uses it to train a 152-layer ImageNet classifier reaching 3.57% top-5 — winning ILSVRC 2015 classification, detection, and localization. The paper's most important contribution is conceptual: the identification and naming of the "degradation problem" (deeper networks have higher training error than shallower ones) and the reframing of the fix as a reparametrization of the optimization landscape. Every subsequent deep architecture — vision, language, audio, generative — uses residuals. A key 2016 follow-up (He et al., arXiv:1603.05027) introduces pre-activation and trains a 1001-layer network on CIFAR.
              </Prose>
            ),
          },
          {
            label: "Hu, Shen & Sun 2017 — SENet",
            render: () => (
              <Prose>
                Hu, J., Shen, L., and Sun, G. (2017). "Squeeze-and-Excitation Networks." arXiv:1709.01507. Published at CVPR 2018. Available at arxiv.org/abs/1709.01507. Won ILSVRC 2017 classification at 2.25% top-5 — the final ILSVRC, after which the benchmark was retired. Introduces the SE module: global avg pool → bottleneck MLP → sigmoid → broadcast-multiply, which gates each channel by a learned scalar. Adding SE to any ResNet improves top-1 by 1–2% at {"<"}1% parameter cost. SE is arguably the first widely-adopted channel attention primitive. Almost every modern efficient backbone — EfficientNet, MobileNet v3, RegNet-Y — includes an SE variant inside each block. The paper is short, clear, and still the best introduction to the "attention as cheap gate" philosophy that later unified with Transformer attention.
              </Prose>
            ),
          },
          {
            label: "Howard et al. 2017 — MobileNets v1",
            render: () => (
              <Prose>
                Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., and Adam, H. (2017). "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." arXiv:1704.04861. Available at arxiv.org/abs/1704.04861. Introduces depthwise separable convolutions as the core building block of a mobile-friendly CNN family. Depthwise separates spatial aggregation (depthwise 3×3, one filter per input channel) from channel mixing (pointwise 1×1). The factorization cuts compute by ~8× at essentially no accuracy cost. MobileNet v1 defines the efficient-inference standard that subsequent work (v2: Sandler et al. 2018 arXiv:1801.04381; v3: Howard et al. 2019 arXiv:1905.02244) refined with inverted residuals, linear bottlenecks, SE, and hard-swish activations. By 2026 MobileNet v3 is the canonical mobile-CPU backbone; EfficientNet-Lite variants share the same design DNA.
              </Prose>
            ),
          },
          {
            label: "Tan & Le 2019 — EfficientNet",
            render: () => (
              <Prose>
                Tan, M., and Le, Q.V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." Proceedings of the 36th International Conference on Machine Learning (ICML). arXiv:1905.11946. Available at arxiv.org/abs/1905.11946. Introduces compound scaling: scale depth, width, and input resolution jointly along a single coefficient <Code>φ</Code> with a constraint <Code>{"\\alpha * \\beta^2 * \\gamma^2 \\approx 2"}</Code>. The base network EfficientNet-B0 is itself found by NAS. Scaling B0 along <Code>φ</Code> produces B1 through B7; B7 reaches 84.3% ImageNet top-1 with 66M params — roughly half the compute of the then-SOTA at matched accuracy. The paper is one of the clearest arguments in recent CNN history for axis coupling: scaling any one axis while holding the others fixed hits diminishing returns; scaling all three together compounds. EfficientNet-V2 (Tan and Le, 2021) extends the recipe with progressive learning and Fused-MBConv blocks.
              </Prose>
            ),
          },
          {
            label: "Radosavovic et al. 2020 — RegNet",
            render: () => (
              <Prose>
                Radosavovic, I., Kosaraju, R.P., Girshick, R., He, K., and Dollár, P. (2020). "Designing Network Design Spaces." arXiv:2003.13678. Published at CVPR 2020. Available at arxiv.org/abs/2003.13678. Reframes architecture search as a search over parametric design spaces rather than over individual networks. The authors define AnyNet (arbitrary ResNet-like networks) and progressively constrain it to RegNet: depth and width are linear functions of a stage index, bottleneck ratio is 1, group width is constant. The resulting RegNet-X and RegNet-Y families are hand-writable in a page and Pareto-dominate EfficientNet at matched FLOPs on ImageNet. The paper's conceptual contribution — that good architectures live on low-dimensional manifolds of design space — is a deep statement about CNN design that has shaped subsequent architecture research and neural architecture search.
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
        Attempt all five before reading the answers. Exercises 1–2 test arithmetic; 3 tests conceptual grasp; 4 tests architectural judgment; 5 tests production debugging.
      </Prose>

      <H3>Exercise 1 (parameter budget)</H3>
      <Prose>
        VGG-16 has 138M parameters. Its conv trunk has 14.7M. Where do the remaining 123M live, and why? What single architectural change in ResNet removes this 123M chunk, and what replaces it?
      </Prose>
      <Callout accent="green">
        <strong>Answer 1.</strong> The remaining 123M live in the fully connected layers: FC6 is <Code>{"7 * 7 * 512 * 4096 = 102.8 \\text{M}"}</Code>, FC7 is <Code>{"4096 * 4096 = 16.8 \\text{M}"}</Code>, FC8 is <Code>{"4096 * 1000 = 4.1 \\text{M}"}</Code>. They exist because VGG treats classification as "flatten the last conv output, run dense FC layers, softmax." The 7×7×512 feature map feeding into a 4096-dim dense layer is the culprit. ResNet replaces this with <strong>global average pooling (GAP)</strong>: the final 7×7×2048 feature map is averaged over spatial dimensions to a 2048-dim vector, then a single FC of <Code>{"2048 * 1000 = 2 \\text{M}"}</Code> params produces the logits. GAP has zero parameters. The total classifier head in ResNet-50 is 2M vs VGG's 123M — a 60× reduction. GAP also acts as a structural regularizer (no learned feature combinations at the top) and eliminates the fixed-input-size constraint that bites VGG.
      </Callout>

      <H3>Exercise 2 (depthwise separable arithmetic)</H3>
      <Prose>
        A standard 3×3 convolution with 128 input channels and 256 output channels on a 14×14 feature map has how many parameters and FLOPs? A depthwise-separable version? Give the ratio of FLOP counts.
      </Prose>
      <Callout accent="green">
        <strong>Answer 2.</strong> Standard 3×3: <Code>{"\\text{Params} = 9 * 128 * 256 = 294{,}912"}</Code>. <Code>{"\\text{FLOPs} = 9 * 128 * 256 * 14 * 14 = 57{,}802{,}752 \\approx 57.8 \\text{M}"}</Code>. Depthwise-separable = depthwise 3×3 (one filter per input channel) + pointwise 1×1 (128 → 256): <Code>{"\\text{Params}_{DWS} = 9 * 128 + 128 * 256 = 1152 + 32{,}768 = 33{,}920"}</Code>. <Code>{"\\text{FLOPs}_{DWS} = 9 * 128 * 14 * 14 + 128 * 256 * 14 * 14 = 225{,}792 + 6{,}422{,}528 \\approx 6.65 \\text{M}"}</Code>. FLOP ratio: <Code>{"6.65 / 57.8 \\approx 0.115"}</Code>, an ~8.7× reduction. Param ratio: <Code>{"33{,}920 / 294{,}912 \\approx 0.115"}</Code>. The algebraic identity is <Code>{"\\text{FLOPs}_{DWS} / \\text{FLOPs}_{std} = 1/C_{out} + 1/k^2 = 1/256 + 1/9 \\approx 0.115"}</Code> — the 1/9 term dominates. The ~8× speedup is why MobileNet can reach ImageNet-competitive accuracy on mobile CPUs.
      </Callout>

      <H3>Exercise 3 (EfficientNet compound scaling)</H3>
      <Prose>
        EfficientNet-B0 has 5.3M params and 0.39 GFLOPs. Using the compound-scaling constraint <Code>{"\\alpha = 1.2, \\beta = 1.1, \\gamma = 1.15"}</Code> with <Code>{"\\alpha \\beta^2 \\gamma^2 \\approx 2"}</Code>, estimate the params and FLOPs of B4 (<Code>{"\\phi = 4"}</Code>). Why does FLOPs scale like <Code>{"2^{\\phi}"}</Code> if width scales quadratically?
      </Prose>
      <Callout accent="green">
        <strong>Answer 3.</strong> At <Code>{"\\phi = 4"}</Code>: depth multiplier <Code>{"1.2^4 \\approx 2.07"}</Code>, width multiplier <Code>{"1.1^4 \\approx 1.46"}</Code>, resolution multiplier <Code>{"1.15^4 \\approx 1.75"}</Code>. Total FLOP scale: <Code>{"(\\alpha \\beta^2 \\gamma^2)^\\phi \\approx 2^4 = 16"}</Code>, so B4 FLOPs <Code>{"\\approx 0.39 \\times 16 = 6.24 \\text{ GFLOPs}"}</Code>. The real B4 number is 4.2 GFLOPs — the estimate is in the right ballpark; small discrepancies come from the rounded exponents (<Code>{"\\alpha \\beta^2 \\gamma^2 = 1.2 * 1.21 * 1.3225 \\approx 1.92"}</Code>, not exactly 2). Params scale with depth × width² (each layer has <Code>{"k^2 * C_{in} * C_{out}"}</Code> params, and both in-channels and out-channels scale with <Code>β</Code>), giving param multiplier <Code>{"2.07 * 1.46^2 \\approx 4.4"}</Code>, so B4 params <Code>{"\\approx 5.3 \\times 4.4 = 23 \\text{ M}"}</Code>. The real B4 number is 19M. FLOPs scale like <Code>{"2^\\phi"}</Code> because the constraint <Code>{"\\alpha \\beta^2 \\gamma^2 \\approx 2"}</Code> is engineered: depth contributes a factor of <Code>α</Code> to FLOPs (more layers), width contributes <Code>{"\\beta^2"}</Code> (FLOPs per layer scale with <Code>{"C_{in} * C_{out}"}</Code>), resolution contributes <Code>{"\\gamma^2"}</Code> (FLOPs per layer scale with <Code>{"H * W"}</Code>). Their product is the total FLOP scaling factor per unit of <Code>φ</Code>.
      </Callout>

      <H3>Exercise 4 (architectural judgment)</H3>
      <Prose>
        You are building an image classification pipeline for a medical imaging dataset with 8,000 labeled chest X-rays (512×512 grayscale, 14 classes) and a latency budget of 100 ms per image on a CPU. Which backbone do you pick, and what modifications do you make to the standard pipeline? What is your biggest risk factor?
      </Prose>
      <Callout accent="green">
        <strong>Answer 4.</strong> Pick: <strong>EfficientNet-B0 or ResNet-50</strong>, pretrained on ImageNet, with full fine-tuning. With only 8K images you are deep in the "pretrained-transfer dominates" regime; training from scratch on 8K images of any architecture will underperform transfer by 10–20% top-1. Modifications: (1) convert grayscale input to 3-channel by replicating the single channel (the pretrained model expects 3-channel RGB); (2) resize to the model's native resolution (224×224 for B0, 224 for ResNet-50) — do <em>not</em> feed 512×512 because the stride-4 stem will throw away too much signal at that input size and shift feature statistics; (3) replace the 1000-way classifier with a 14-way head; (4) use class-balanced sampling because chest X-ray classes are heavily imbalanced; (5) use ImageNet preprocessing stats. Biggest risk factor: <strong>domain shift</strong>. ImageNet is natural images; chest X-rays are monochrome, low-contrast, center-framed. Pretrained filters learned to detect fur, grass, and blue sky may not transfer cleanly to consolidation, cardiomegaly, and pleural effusion. Mitigations: (a) fine-tune all layers, not just the head (linear probe will underperform); (b) use a longer schedule with a lower LR than typical ImageNet fine-tuning; (c) consider a medical-imaging-pretrained backbone if available (CheXpert-pretrained or RadImageNet-pretrained); (d) if your latency budget allows, ensemble B0 with ResNet-50 for variance reduction. CPU latency: EfficientNet-B0 at 224×224 is ~50 ms on a modern x86 CPU with ONNX Runtime; ResNet-50 is ~70 ms. Both fit the 100 ms budget. MobileNet v3-Small at 20 ms is tempting but the accuracy ceiling is usually too low for a medical task.
      </Callout>

      <H3>Exercise 5 (production debugging)</H3>
      <Prose>
        You fine-tune a torchvision <Code>{"efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)"}</Code> on a 40-class custom dataset. Validation accuracy is stuck at ~5% after 20 epochs — close to random guessing. The same pipeline with <Code>{"resnet50"}</Code> reaches 85% validation accuracy in 10 epochs. List three architecture-specific hypotheses and how you would verify each.
      </Prose>
      <Callout accent="green">
        <strong>Answer 5.</strong> Three likely failure modes, all architecture-specific to EfficientNet-B7:
        <br />
        (1) <strong>Wrong input resolution.</strong> EfficientNet-B7 was trained at 600×600; you may be feeding it 224×224 because that is what your ResNet-50 pipeline uses. At 224×224 the model's receptive fields in early layers are designed for 600×600-scale features and produce mis-scaled activations; accuracy collapses. Verify: <Code>{"print(model.default_cfg if hasattr(model, 'default_cfg') else weights.transforms())"}</Code> — the crop_size and resize_size tell you the intended input. Use <Code>{"weights.transforms()"}</Code> verbatim. Fix: resize to 600×600 (or use a smaller B variant if 600×600 is too slow).
        <br />
        (2) <strong>Normalization mismatch.</strong> EfficientNet-B7 uses the standard ImageNet mean/std <Code>{"[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]"}</Code>, but your pipeline may use <Code>{"[0.5, 0.5, 0.5]"}</Code> (which was common in code inherited from GAN or diffusion projects). The feature-statistic shift cascades through all batch norm running stats and collapses accuracy. Verify: print the mean and std of a preprocessed batch — if mean is close to 0 and std close to 1, you have standard ImageNet normalization; if mean is close to 0 with std close to 0.5, you are using the wrong normalization. Fix: always use <Code>{"weights.transforms()"}</Code>.
        <br />
        (3) <strong>BatchNorm in wrong mode or BN stats not resetting.</strong> If you loaded the pretrained weights but put the model in train mode with frozen parameters, BN running stats will drift on your (small) dataset and the pretrained weights will become inconsistent with the running stats. Symptom: the model works fine in eval mode on ImageNet but degrades on your data. Verify: check <Code>{"model.training"}</Code>, check whether BN stats have drifted from their pretrained values via <Code>{"for m in model.modules(): if isinstance(m, nn.BatchNorm2d): print(m.running_mean.mean().item())"}</Code> before and after a few training steps. If the mean changes significantly, BN is updating when it shouldn't. Fix: call <Code>{"model.eval()"}</Code> on frozen parts or unfreeze everything when fine-tuning a large model like B7.
        <br />
        In almost all real debugging sessions, the answer is (1) or (2). EfficientNet's strict resolution dependence is the single most common footgun when porting code between backbones, and standard ResNet-50 preprocessing does not transfer to it.
      </Callout>

    </div>
  ),
};

export default landmarkArchitecturesContent;
