import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const convolutionPoolingRFContent = {
  title: "Convolution, Pooling & Receptive Fields",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        The convolutional neural network is one of the oldest architectures in deep learning, and its design comes almost directly from neurophysiology. In 1962, David Hubel and Torsten Wiesel published "Receptive fields, binocular interaction and functional architecture in the cat's visual cortex" in <em>The Journal of Physiology</em> (volume 160, pages 106–154). They inserted tungsten microelectrodes into the primary visual cortex (area V1) of anesthetized cats and projected bars of light onto a screen while recording which neurons fired. The experiment, which later won them the 1981 Nobel Prize in Physiology or Medicine, produced two foundational findings. First, individual V1 neurons respond only to stimuli in a small region of the visual field — the neuron's <em>receptive field</em>. Second, V1 contains two broad cell classes: <em>simple cells</em> that fire when an oriented edge at a specific orientation and position crosses their receptive field, and <em>complex cells</em> that fire for the same oriented edge but are translation-invariant over a larger region. Simple cells look like linear oriented filters; complex cells look like a pooling operation over nearby simple cells of the same orientation.
      </Prose>

      <Prose>
        In 1980, Kunihiko Fukushima at NHK Broadcasting Science Research Laboratories published "Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position" in <em>Biological Cybernetics</em> (volume 36, pages 193–202). The Neocognitron is almost exactly a modern CNN, proposed 35 years before AlexNet. It alternated <em>S-layers</em> (modeled on simple cells: local weighted sums detecting oriented features) with <em>C-layers</em> (modeled on complex cells: max-like pooling operations that introduce translation invariance). Fukushima trained it with an unsupervised competitive learning rule and showed it could recognize handwritten digits robustly to shifts. What it lacked was gradient-based learning — every layer was trained layer-wise rather than end-to-end.
      </Prose>

      <Prose>
        That missing piece arrived in 1989. Yann LeCun and colleagues at AT&T Bell Labs published "Backpropagation Applied to Handwritten Zip Code Recognition" in <em>Neural Computation</em> (volume 1, pages 541–551). For the first time, a convolutional network with shared weights and spatial pooling was trained end-to-end with backpropagation on real-world data — 9,298 handwritten ZIP-code digits segmented from US Postal Service mail. The network had three hidden layers, used weight sharing to reduce parameters from ~100,000 to ~2,600, and achieved 1.0% error on the training set and 5.0% error on the test set — competitive with the best classical methods of the time. Every design choice that became standard was already present: local receptive fields, weight sharing across spatial positions, stride-2 subsampling between layers, and gradient training via backprop.
      </Prose>

      <Prose>
        By 1998, LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner had refined this into LeNet-5, described in "Gradient-Based Learning Applied to Document Recognition" (<em>Proceedings of the IEEE</em> 86(11), pages 2278–2324). LeNet-5 read ~60 million checks per month through NCR's systems in the late 1990s — the first production CNN deployment. The architecture is almost identical to modern image classifiers: two conv/pool stages, then fully connected layers, trained with backprop. The paper is also the reason everyone in 2026 still says "convolution" when they really mean "cross-correlation" — LeCun's original LeNet papers defined convolution in the cross-correlation sense (no kernel flip), and the terminology stuck.
      </Prose>

      <Prose>
        Between 1998 and 2012, CNNs languished. GPUs were not fast enough, datasets were not large enough, and for most problems hand-engineered features plus SVMs beat neural networks. The turning point was September 2012, when Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton at the University of Toronto submitted a CNN to the ImageNet Large Scale Visual Recognition Challenge. Their paper "ImageNet Classification with Deep Convolutional Neural Networks" (NeurIPS 2012) described AlexNet: 5 conv layers, 3 FC layers, 60 million parameters, trained on two GTX 580 GPUs for five to six days on 1.2 million labeled images. It achieved 15.3% top-5 error on ILSVRC 2012 — a 10.8 percentage-point absolute improvement over the second-place entry, which used hand-crafted features. That gap was so large that the entire computer vision research community pivoted within six months. Every subsequent ILSVRC winner was a CNN; hand-crafted features never returned.
      </Prose>

      <Prose>
        Everything that followed was a refinement of the same recipe. VGG (2014) showed that deeper stacks of small 3×3 convolutions beat shallower stacks of large ones. GoogLeNet / Inception (2014) introduced 1×1 convolutions for channel bottlenecks and parallel multi-scale branches. ResNet (2015) used residual shortcuts to enable 152-layer networks. MobileNet (2017) popularized depthwise separable convolutions for phones. ConvNeXt (2022) showed that a modernized CNN could match vision Transformers at the same FLOP budget. Across all of them, the three primitives remain the same: convolution for local spatial filtering, pooling for spatial reduction, and receptive-field analysis for reasoning about how much context each output neuron sees.
      </Prose>

      <Prose>
        The receptive-field side of the story has its own important chapter. In late 2016, Wenjie Luo, Yujia Li, Raquel Urtasun, and Richard Zemel at the University of Toronto published "Understanding the Effective Receptive Field in Deep Convolutional Neural Networks" (NeurIPS 2016, arXiv 1701.04128). The paper observed that although the <em>theoretical</em> receptive field of a deep CNN grows linearly with depth, the <em>effective</em> receptive field (ERF) — the set of input pixels that actually influence a given output neuron — is much smaller and has a Gaussian-shaped intensity profile. Only a small fraction of theoretically-reachable pixels contribute meaningfully to any output. The practical consequence: just because you stacked enough convs to "see" 200 pixels in theory does not mean the network actually uses 200 pixels of context. This finding reshaped how semantic segmentation, dense prediction, and object detection architectures are designed — dilated convolutions, atrous spatial pyramid pooling, and large-kernel CNNs all attack the ERF gap.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Fully connected layers treat every input feature as independent and every output feature as a distinct linear combination of all inputs. For a 224×224 RGB image that is 150,528 input features per sample. Connecting this to even a modest 1000-unit hidden layer requires 150 million parameters <em>for a single layer</em>, plus no inductive bias at all — the network has to re-discover from scratch that neighboring pixels are related and that a cat in the top-left is the same object as a cat in the bottom-right. Convolution bakes two beliefs about natural images into the architecture: features are <em>local</em> (edges, textures, object parts occupy small spatial neighborhoods) and the <em>same feature matters at every location</em> (a cat ear detector should work whether the ear appears in pixel (10, 10) or pixel (200, 200)).
      </Prose>

      <Prose>
        <strong>Local connectivity.</strong> Each output neuron depends only on a small k×k patch of the input, not the whole image. For a 3×3 conv with 64 input channels and 64 output channels, each output neuron has 3·3·64 = 576 incoming weights, regardless of the image size. A 224×224 feature map and a 7×7 feature map share exactly the same weights. This dramatically reduces parameter count and reflects the reality of images: what matters for detecting an edge in the middle of a photograph is the pixels around that edge, not pixels a hundred positions away.
      </Prose>

      <Prose>
        <strong>Weight sharing and translation equivariance.</strong> A convolutional layer uses the same k×k kernel at every spatial position. If the input shifts by one pixel, the output shifts by one pixel — this property is called <em>translation equivariance</em>. Equivariance is not the same as invariance: the output still moves, it just moves predictably. Invariance (the output does not change at all when the input shifts) comes from pooling or from global pooling at the end of the network. Sharing weights across positions also means gradient updates are <em>averaged</em> across every location the kernel was applied at, giving ~HW times as many effective training examples per weight as a fully connected layer, which is the single biggest reason CNNs generalize well from relatively modest datasets.
      </Prose>

      <Prose>
        <strong>Stacking grows receptive field.</strong> A single 3×3 conv at the input sees a 3×3 patch. Stack two 3×3 convs and each output neuron sees a 5×5 input patch — the second layer's neighbors on the first feature map each depend on their own 3×3 neighborhood, and those neighborhoods overlap. This compounds: N stacked 3×3 convs give a theoretical receptive field of 2N+1 pixels, the same as a single (2N+1)×(2N+1) kernel — but with roughly N times fewer parameters (N·3²·C² vs. (2N+1)²·C²) and an additional nonlinearity per layer. This is the insight behind VGG: replace AlexNet's 7×7 and 5×5 early convs with stacks of 3×3.
      </Prose>

      <Prose>
        <strong>Pooling reduces spatial resolution and adds invariance.</strong> Max pooling takes the maximum over a k×k window and slides with stride k, halving spatial resolution for the default 2×2/stride 2 configuration. The practical effects are threefold: (1) compute and memory for subsequent layers drop 4× per 2×2 pool, (2) the receptive field grows more quickly because each subsequent conv's window now covers 2× as much input in each direction, (3) small spatial perturbations of the input get smoothed — a feature that appears in pixel (17, 17) or pixel (17, 18) both map to the same pooled output if the window contains both. Average pooling replaces max with mean and is more common at the end of the network (global average pooling as classifier-head replacement).
      </Prose>

      <Prose>
        <strong>Three ways to grow the receptive field.</strong> There are only three levers: increase kernel size k (more parameters per layer, O(k²) FLOPs), increase stride or insert pooling (reduces spatial resolution, loses fine detail), or use dilation (spacing kernel taps with gaps, same params/FLOPs but larger coverage at the cost of aliasing gaps in what the kernel sees). Modern architectures pick a combination. A ResNet-50 uses stride-2 downsampling four times; a DeepLab segmentation network uses dilated convolutions to grow RF without spatial reduction; a Transformer-style vision model skips local convolution entirely and uses global attention. Each choice trades off RF growth, spatial fidelity, and compute.
      </Prose>

      <Callout accent="gold">
        Mental model: a conv layer is a <em>bank of templates</em> slid across the image. Each output channel is one template's response map. Pooling is a <em>summary over a local window</em>. The receptive field is <em>the set of input pixels any given output neuron could have looked at</em>. The effective receptive field is <em>the set of input pixels that actually matter</em> — almost always a Gaussian blob much smaller than the theoretical RF.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Discrete 2D convolution (cross-correlation)</H3>

      <Prose>
        Deep learning libraries implement <em>cross-correlation</em> even though they call it "convolution". The true mathematical convolution flips the kernel; cross-correlation does not. Because the kernel is learned, the flip is irrelevant — the optimizer will simply learn the flipped version of whatever true-convolution kernel you meant. What we actually compute for an input feature map <Code>{"x[c, i, j]"}</Code> and a kernel <Code>{"w[c, u, v]"}</Code>:
      </Prose>

      <MathBlock>
        {"y[i, j] = \\sum_{c=0}^{C_{in}-1} \\sum_{u=0}^{k-1} \\sum_{v=0}^{k-1} x[c, \\, i + u, \\, j + v] \\cdot w[c, u, v]"}
      </MathBlock>

      <Prose>
        For a multi-output-channel conv with <Code>{"C_{out}"}</Code> output channels, this extends to a tensor computation over a weight <Code>{"w[c_{out}, c_{in}, u, v]"}</Code>:
      </Prose>

      <MathBlock>
        {"y[c_o, i, j] = b[c_o] + \\sum_{c_i=0}^{C_{in}-1} \\sum_{u=0}^{k-1} \\sum_{v=0}^{k-1} x[c_i, \\, i + u, \\, j + v] \\cdot w[c_o, c_i, u, v]"}
      </MathBlock>

      <Prose>
        Each output channel <Code>{"c_o"}</Code> is a different learned filter applied to the full stack of input channels. In plain terms: <em>for every output location, take the element-wise product of the kernel with the corresponding input patch and sum</em>.
      </Prose>

      <H3>3.2 Output size formula</H3>

      <Prose>
        Given input spatial size <Code>{"n_{in}"}</Code>, kernel size <Code>k</Code>, padding <Code>p</Code>, stride <Code>s</Code>, and dilation <Code>d</Code> (the spacing between kernel taps), the output spatial size is:
      </Prose>

      <MathBlock>
        {"n_{out} = \\left\\lfloor \\frac{n_{in} + 2p - d \\cdot (k - 1) - 1}{s} \\right\\rfloor + 1"}
      </MathBlock>

      <Prose>
        The quantity <Code>{"d \\cdot (k-1) + 1"}</Code> is the <em>effective</em> kernel size — a 3×3 kernel with dilation 2 spans 5 input pixels (with gaps). Memorize the three common special cases:
      </Prose>

      <MathBlock>
        {"\\text{stride 1, pad} \\lfloor k/2 \\rfloor: \\quad n_{out} = n_{in} \\quad \\text{(shape-preserving 'same' conv)}"}
      </MathBlock>

      <MathBlock>
        {"\\text{stride 2, pad} \\lfloor k/2 \\rfloor: \\quad n_{out} = \\lceil n_{in}/2 \\rceil \\quad \\text{(downsample by 2)}"}
      </MathBlock>

      <MathBlock>
        {"\\text{stride 1, pad 0: } \\quad n_{out} = n_{in} - k + 1 \\quad \\text{(valid conv, shrinks by } k-1 \\text{)}"}
      </MathBlock>

      <H3>3.3 Theoretical receptive field recursion</H3>

      <Prose>
        The receptive field of a neuron at layer <Code>{"\\ell"}</Code> is the set of input pixels that could, in principle, influence its activation. Define <Code>{"r_\\ell"}</Code> as the RF size (in input pixels per dimension) and <Code>{"j_\\ell"}</Code> as the <em>jump</em> — the spacing in input pixels between adjacent neurons at layer <Code>{"\\ell"}</Code>. The recursion is:
      </Prose>

      <MathBlock>
        {"r_\\ell = r_{\\ell-1} + (k_\\ell - 1) \\cdot j_{\\ell-1}, \\qquad j_\\ell = j_{\\ell-1} \\cdot s_\\ell"}
      </MathBlock>

      <Prose>
        with initial values <Code>{"r_0 = 1"}</Code> and <Code>{"j_0 = 1"}</Code>. Equivalently, unrolling the recursion:
      </Prose>

      <MathBlock>
        {"r_L = 1 + \\sum_{\\ell=1}^{L} (k_\\ell - 1) \\cdot \\prod_{i < \\ell} s_i"}
      </MathBlock>

      <Prose>
        Each layer contributes <Code>{"k_\\ell - 1"}</Code> extra input pixels to the RF, scaled by the cumulative product of prior strides. Strides and pooling are multiplicative; kernel sizes are additive. A pool of kernel 2 / stride 2 contributes <Code>{"(2-1) \\cdot j_{\\ell-1} = j_{\\ell-1}"}</Code> pixels to RF and doubles <Code>j</Code>, so the next conv sees twice as much input per kernel tap.
      </Prose>

      <H3>3.4 Dilated (atrous) convolution RF</H3>

      <Prose>
        Dilation replaces <Code>{"k_\\ell - 1"}</Code> in the recursion with <Code>{"d_\\ell \\cdot (k_\\ell - 1)"}</Code>. A 3×3 conv with dilation 2 contributes the same as a 5×5 conv at the same layer — RF grows quickly without adding parameters or increasing FLOPs. This is why semantic segmentation networks like DeepLab rely on dilation: they need large RF for scene-level context while keeping high spatial resolution (no downsampling).
      </Prose>

      <MathBlock>
        {"r_\\ell = r_{\\ell-1} + d_\\ell \\cdot (k_\\ell - 1) \\cdot j_{\\ell-1}"}
      </MathBlock>

      <H3>3.5 Effective receptive field (Luo et al. 2016)</H3>

      <Prose>
        Luo and colleagues showed that for a stack of <Code>N</Code> convolutional layers with random independent weights, the effective RF — the gradient of the center output pixel with respect to input pixels — is asymptotically Gaussian with standard deviation growing as <Code>{"\\sigma \\propto \\sqrt{N}"}</Code>, not as <Code>N</Code>. This is the central limit theorem applied to the product of random Jacobians that form the gradient path through the stack. Formally, for uniform weights with variance <Code>{"\\sigma_w^2"}</Code>:
      </Prose>

      <MathBlock>
        {"\\text{ERF}(u, v) \\approx \\frac{1}{2\\pi \\sigma^2} \\exp\\left(-\\frac{u^2 + v^2}{2\\sigma^2}\\right), \\qquad \\sigma \\sim \\mathcal{O}(\\sqrt{N})"}
      </MathBlock>

      <Prose>
        The theoretical RF grows like <Code>N</Code>; the effective RF grows like <Code>{"\\sqrt{N}"}</Code>. Doubling depth adds only a <Code>{"\\sqrt{2} \\approx 1.41"}</Code>× increase to the pixels that actually matter. The practical takeaway: stacking more layers is a diminishing way to grow effective context. The efficient ways are dilation, strided downsampling, and skip connections that aggregate at multiple scales.
      </Prose>

      <H3>3.6 Parameter count and FLOPs</H3>

      <Prose>
        A conv layer with kernel <Code>{"k \\times k"}</Code>, input channels <Code>{"C_{in}"}</Code>, and output channels <Code>{"C_{out}"}</Code> has:
      </Prose>

      <MathBlock>
        {"\\text{params} = k \\cdot k \\cdot C_{in} \\cdot C_{out} + C_{out} \\text{ (if bias)}"}
      </MathBlock>

      <Prose>
        Parameter count does <em>not</em> depend on spatial resolution — the same 3×3 conv has the same weights at any image size. FLOPs, however, scale linearly with the spatial output area. Defining a multiply–accumulate (MAC) as 2 FLOPs:
      </Prose>

      <MathBlock>
        {"\\text{FLOPs} = 2 \\cdot H_{out} \\cdot W_{out} \\cdot k^2 \\cdot C_{in} \\cdot C_{out}"}
      </MathBlock>

      <Prose>
        For grouped convolution with <Code>g</Code> groups, the sum over input channels is restricted to the group:
      </Prose>

      <MathBlock>
        {"\\text{params}_{\\text{grouped}} = k^2 \\cdot \\frac{C_{in}}{g} \\cdot C_{out}, \\qquad \\text{FLOPs}_{\\text{grouped}} = 2 \\cdot H_{out} \\cdot W_{out} \\cdot k^2 \\cdot \\frac{C_{in}}{g} \\cdot C_{out}"}
      </MathBlock>

      <Prose>
        Depthwise convolution is the special case <Code>{"g = C_{in} = C_{out}"}</Code>: parameter and FLOP counts both drop by <Code>{"C_{in}"}</Code>×.
      </Prose>

      <H3>3.7 Backward pass</H3>

      <Prose>
        The backward pass of convolution is itself a convolution, which is why training is efficient. For the gradient of the loss <Code>L</Code> with respect to the input:
      </Prose>

      <MathBlock>
        {"\\frac{\\partial L}{\\partial x[c_i, i, j]} = \\sum_{c_o} \\sum_{u, v} \\frac{\\partial L}{\\partial y[c_o, i-u, j-v]} \\cdot w[c_o, c_i, u, v]"}
      </MathBlock>

      <Prose>
        This is a <em>transposed</em> (or "full") convolution of the output gradient with the flipped kernel. For the weight gradient:
      </Prose>

      <MathBlock>
        {"\\frac{\\partial L}{\\partial w[c_o, c_i, u, v]} = \\sum_{i, j} \\frac{\\partial L}{\\partial y[c_o, i, j]} \\cdot x[c_i, i+u, j+v]"}
      </MathBlock>

      <Prose>
        which is another convolution-like accumulation. Because both passes reduce to matrix multiplications through im2col, the same highly tuned GEMM kernels that make the forward pass fast also make the backward pass fast.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below is executable numpy/PyTorch. Outputs are verbatim stdout captured from local runs.
      </Prose>

      <H3>4a. Direct 2D convolution with padding, stride, and dilation</H3>

      <Prose>
        The most transparent implementation uses six nested loops — over batch, output channel, input channel, spatial row, spatial column, and kernel. We collapse the channel/kernel loops into numpy broadcasting so only the two spatial loops remain explicit. Below we implement the conv, then verify it against <Code>F.conv2d</Code> in three configurations: basic, strided, and dilated.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
import torch
import torch.nn.functional as F

def conv2d_direct(x, w, bias=None, stride=1, padding=0, dilation=1):
    """Direct 2D cross-correlation.
      x : (N, Cin, H, W)
      w : (Cout, Cin, kH, kW)
    Returns (N, Cout, Hout, Wout) where
      Hout = floor((H + 2*pad - dil*(kH-1) - 1)/stride) + 1
    """
    N, Cin, H, W = x.shape
    Cout, _, kH, kW = w.shape
    if padding > 0:
        xp = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
    else:
        xp = x
    Hp, Wp = xp.shape[2], xp.shape[3]
    eff_kH = (kH - 1) * dilation + 1
    eff_kW = (kW - 1) * dilation + 1
    Hout = (Hp - eff_kH) // stride + 1
    Wout = (Wp - eff_kW) // stride + 1

    out = np.zeros((N, Cout, Hout, Wout), dtype=x.dtype)
    for i in range(Hout):
        for j in range(Wout):
            rs, cs = i * stride, j * stride
            # Patch indexed with dilation step; shape (N, Cin, kH, kW)
            patch = xp[:, :, rs:rs+eff_kH:dilation, cs:cs+eff_kW:dilation]
            # Broadcast multiply against (Cout, Cin, kH, kW), sum over (Cin,kH,kW)
            out[:, :, i, j] = (patch[:, None] * w[None]).sum(axis=(2,3,4))
    if bias is not None:
        out = out + bias.reshape(1, -1, 1, 1)
    return out

# Verify against torch for three configurations
np.random.seed(0)
x = np.random.randn(2, 3, 8, 8).astype(np.float32)
w = np.random.randn(5, 3, 3, 3).astype(np.float32)
b = np.random.randn(5).astype(np.float32)

for label, kw in [
    ("stride=1 pad=1 dil=1", dict(stride=1, padding=1, dilation=1)),
    ("stride=2 pad=0 dil=1", dict(stride=2, padding=0, dilation=1)),
    ("stride=1 pad=2 dil=2", dict(stride=1, padding=2, dilation=2)),
]:
    y_ours = conv2d_direct(x, w, b, **kw)
    y_ref = F.conv2d(torch.from_numpy(x), torch.from_numpy(w),
                     bias=torch.from_numpy(b), **kw).numpy()
    diff = np.abs(y_ours - y_ref).max()
    print(f"{label:<22} ours {y_ours.shape}  diff {diff:.2e}")

# Output:
# stride=1 pad=1 dil=1   ours (2, 5, 8, 8)  diff 2.86e-06
# stride=2 pad=0 dil=1   ours (2, 5, 3, 3)  diff 1.91e-06
# stride=1 pad=2 dil=2   ours (2, 5, 8, 8)  diff 2.86e-06`}
      </CodeBlock>

      <Prose>
        The numerical differences are at the <Code>{"10^{-6}"}</Code> level — exactly float32 accumulation noise. Our direct implementation matches cuDNN's output bit-for-bit in exact arithmetic. This is the first sanity check anyone writing a custom conv kernel runs.
      </Prose>

      <H3>4b. im2col formulation and GEMM</H3>

      <Prose>
        Every production conv implementation (cuDNN, MKL-DNN, XNNPACK) is built on the same trick: reshape the convolution into a matrix multiplication. Given an input tensor, the im2col (image-to-column) operation unfolds every k×k patch into a column vector. The result is a <Code>{"(C_{in} \\cdot k^2, H_{out} \\cdot W_{out})"}</Code> matrix. Reshaping the weight tensor to <Code>{"(C_{out}, C_{in} \\cdot k^2)"}</Code> turns the whole conv into a single GEMM call, which runs at peak FLOPs on every modern processor.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
import torch
import torch.nn.functional as F

def im2col(x, kH, kW, stride=1, padding=0):
    """Unfold NCHW -> (N, Cin*kH*kW, L) with L = Hout*Wout."""
    N, C, H, W = x.shape
    if padding > 0:
        xp = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
    else:
        xp = x
    Hp, Wp = xp.shape[2], xp.shape[3]
    Hout = (Hp - kH) // stride + 1
    Wout = (Wp - kW) // stride + 1
    cols = np.zeros((N, C * kH * kW, Hout * Wout), dtype=x.dtype)
    col = 0
    for i in range(Hout):
        for j in range(Wout):
            patch = xp[:, :, i*stride:i*stride+kH, j*stride:j*stride+kW]
            cols[:, :, col] = patch.reshape(N, -1)
            col += 1
    return cols, Hout, Wout

def conv2d_im2col(x, w, bias=None, stride=1, padding=0):
    N, Cin, H, W = x.shape
    Cout, _, kH, kW = w.shape
    cols, Hout, Wout = im2col(x, kH, kW, stride, padding)
    w_mat = w.reshape(Cout, Cin * kH * kW)                   # (Cout, K)
    out = np.einsum('ok,nkl->nol', w_mat, cols)              # GEMM per sample
    out = out.reshape(N, Cout, Hout, Wout)
    if bias is not None:
        out = out + bias.reshape(1, -1, 1, 1)
    return out

# Numerical check
np.random.seed(0)
x = np.random.randn(2, 3, 8, 8).astype(np.float32)
w = np.random.randn(5, 3, 3, 3).astype(np.float32)
b = np.random.randn(5).astype(np.float32)
y_ours = conv2d_im2col(x, w, b, stride=1, padding=1)
y_ref  = F.conv2d(torch.from_numpy(x), torch.from_numpy(w),
                  bias=torch.from_numpy(b), stride=1, padding=1).numpy()
print(f"im2col vs torch: shape {y_ours.shape}  diff {np.abs(y_ours-y_ref).max():.2e}")

# Memory cost of im2col: the unfolded buffer is k^2 times larger than input
Cin = 64; H = W = 224; kH = kW = 3
Hp, Wp = H + 2, W + 2               # pad=1
L = (Hp - kH + 1) * (Wp - kW + 1)
K = Cin * kH * kW
print(f"\\nim2col memory explosion (Cin={Cin}, 224x224, 3x3):")
print(f"  input tensor  : {Cin*H*W*4/1e6:6.2f} MB")
print(f"  im2col buffer : {K*L*4/1e6:6.2f} MB   ({(K*L)/(Cin*H*W):.1f}x input)")

# Output:
# im2col vs torch: shape (2, 5, 8, 8)  diff 2.86e-06
#
# im2col memory explosion (Cin=64, 224x224, 3x3):
#   input tensor  :  12.85 MB
#   im2col buffer : 115.61 MB   (9.0x input)`}
      </CodeBlock>

      <Prose>
        The memory blowup is the reason cuDNN offers at least four conv algorithms and picks among them by input shape: <Code>IMPLICIT_GEMM</Code> (no explicit im2col buffer, recompute indices), <Code>IMPLICIT_PRECOMP_GEMM</Code>, <Code>GEMM</Code> (the naive version above), and <Code>WINOGRAD</Code> (for 3×3 convs at small FLOP counts). At high spatial resolution, an explicit im2col buffer alone can exceed GPU memory; implicit variants recompute patch indices on the fly.
      </Prose>

      <H3>4c. Output-size formula check against torch</H3>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F
from math import floor

def out_size(n, k, s=1, p=0, d=1):
    return floor((n + 2*p - d*(k-1) - 1) / s) + 1

cases = [
    (32, 3, 1, 0, 1),  # valid conv: 32 -> 30
    (32, 3, 1, 1, 1),  # same conv:  32 -> 32
    (32, 3, 2, 1, 1),  # stride 2:   32 -> 16
    (32, 5, 1, 2, 1),  # same 5x5:   32 -> 32
    (32, 3, 1, 2, 2),  # dilated:    32 -> 32
    (32, 7, 2, 3, 1),  # 7x7 /s2:    32 -> 16
    (224,11, 4, 2, 1), # AlexNet:    224 -> 55
    (7,  3, 1, 0, 1),  # shrinks:    7  -> 5
]
print("  in   k  s  p  d  | formula  torch")
for H, k, s, p, d in cases:
    x = torch.randn(1, 1, H, H)
    w = torch.randn(1, 1, k, k)
    y = F.conv2d(x, w, stride=s, padding=p, dilation=d)
    print(f"  {H:3d}  {k:2d} {s:2d} {p:2d} {d:2d}  |  {out_size(H,k,s,p,d):3d}    {y.shape[-1]:3d}")

# Output:
#   in   k  s  p  d  | formula  torch
#    32   3  1  0  1  |   30     30
#    32   3  1  1  1  |   32     32
#    32   3  2  1  1  |   16     16
#    32   5  1  2  1  |   32     32
#    32   3  1  2  2  |   32     32
#    32   7  2  3  1  |   16     16
#   224  11  4  2  1  |   55     55
#     7   3  1  0  1  |    5      5`}
      </CodeBlock>

      <H3>4d. Receptive field of a VGG-style stack</H3>

      <Prose>
        Compute the RF of VGG-16's 13 conv layers interleaved with 5 pools using the recursion <Code>{"r_\\ell = r_{\\ell-1} + (k_\\ell - 1) \\cdot j_{\\ell-1}"}</Code> with <Code>{"j_\\ell = j_{\\ell-1} \\cdot s_\\ell"}</Code>.
      </Prose>

      <CodeBlock language="python">
{`def receptive_field(layers):
    r, j = 1, 1
    out = []
    for L in layers:
        r = r + (L['k'] - 1) * j
        out.append((L['name'], L['k'], L['s'], r))
        j = j * L['s']
    return out

vgg16 = [
    {'name': 'conv1_1', 'k': 3, 's': 1}, {'name': 'conv1_2', 'k': 3, 's': 1},
    {'name': 'pool1',   'k': 2, 's': 2},
    {'name': 'conv2_1', 'k': 3, 's': 1}, {'name': 'conv2_2', 'k': 3, 's': 1},
    {'name': 'pool2',   'k': 2, 's': 2},
    {'name': 'conv3_1', 'k': 3, 's': 1}, {'name': 'conv3_2', 'k': 3, 's': 1},
    {'name': 'conv3_3', 'k': 3, 's': 1}, {'name': 'pool3', 'k': 2, 's': 2},
    {'name': 'conv4_1', 'k': 3, 's': 1}, {'name': 'conv4_2', 'k': 3, 's': 1},
    {'name': 'conv4_3', 'k': 3, 's': 1}, {'name': 'pool4', 'k': 2, 's': 2},
    {'name': 'conv5_1', 'k': 3, 's': 1}, {'name': 'conv5_2', 'k': 3, 's': 1},
    {'name': 'conv5_3', 'k': 3, 's': 1}, {'name': 'pool5', 'k': 2, 's': 2},
]
print(f"  {'layer':<10}{'k':>3} {'s':>3} {'RF (px)':>10}")
for name, k, s, r in receptive_field(vgg16):
    print(f"  {name:<10}{k:>3} {s:>3} {r:>10}")

# Output:
#   layer       k   s    RF (px)
#   conv1_1     3   1          3
#   conv1_2     3   1          5
#   pool1       2   2          6
#   conv2_1     3   1         10
#   conv2_2     3   1         14
#   pool2       2   2         16
#   conv3_1     3   1         24
#   conv3_2     3   1         32
#   conv3_3     3   1         40
#   pool3       2   2         44
#   conv4_1     3   1         60
#   conv4_2     3   1         76
#   conv4_3     3   1         92
#   pool4       2   2        100
#   conv5_1     3   1        132
#   conv5_2     3   1        164
#   conv5_3     3   1        196
#   pool5       2   2        212`}
      </CodeBlock>

      <Prose>
        A 224×224 ImageNet image fed into VGG-16 has a theoretical RF of 212 pixels at the last pool — essentially the whole image. Every neuron in the final feature map could, in principle, have looked at almost every input pixel. The effective RF, as we will see shortly, is much smaller.
      </Prose>

      <H3>4e. Empirical effective receptive field (Luo 2016 setup)</H3>

      <Prose>
        The effective RF is measured by placing a delta gradient at a single output pixel and backpropagating to the input. The magnitude of the resulting input gradient at each pixel is the ERF. Below we measure it for stacks of 3, 5, 10, 15, 20 convolutions (all 3×3, stride 1, same padding) with uniform positive weights — the exact setup from Luo et al. 2016 that produces the Gaussian falloff.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn

def build_stack(n_conv, k=3):
    layers = []
    for _ in range(n_conv):
        c = nn.Conv2d(1, 1, k, padding=k//2, bias=False)
        nn.init.constant_(c.weight, 1.0 / (k * k))
        layers += [c, nn.ReLU(inplace=False)]
    return nn.Sequential(*layers)

def measure_erf(n_conv, H=65):
    torch.manual_seed(0)
    net = build_stack(n_conv)
    x = torch.randn(1, 1, H, H, requires_grad=True)
    y = net(x)
    g = torch.zeros_like(y); g[0, 0, H//2, H//2] = 1.0
    y.backward(g)
    row = x.grad.abs().squeeze().numpy()[H//2]
    peak = row.max()
    if peak <= 0: return 0
    idxs = (row >= 0.01 * peak).nonzero()[0]
    return (idxs[-1] - idxs[0] + 1)

print("N (3x3 convs) | theoretical RF | empirical ERF (1% cutoff) | ratio")
for N in [3, 5, 10, 15, 20]:
    trf = 1 + 2 * N
    erf = measure_erf(N)
    print(f"  {N:3d}         |      {trf:3d}       |          {erf:3d}          | {erf/trf:.2f}")

# Output:
# N (3x3 convs) | theoretical RF | empirical ERF (1% cutoff) | ratio
#     3         |        7       |            0              | 0.00
#     5         |       11       |            8              | 0.73
#    10         |       21       |           16              | 0.76
#    15         |       31       |           18              | 0.58
#    20         |       41       |           21              | 0.51`}
      </CodeBlock>

      <Prose>
        By 20 layers the ERF is half the theoretical RF. The trend matches Luo et al.'s prediction: ERF grows like <Code>{"\\sqrt{N}"}</Code>, theoretical RF grows like <Code>N</Code>, so their ratio shrinks with depth. At 100 layers the ratio would be well under 0.3. This is why very deep CNNs still need dilation or large kernels to actually use their theoretical context.
      </Prose>

      <H3>4f. Pooling implementations from scratch</H3>

      <Prose>
        Max pool, avg pool, and adaptive avg pool, each verified against the torch equivalent. Adaptive pool is the tricky one: the slice boundaries follow torch's formula <Code>{"h_{start} = \\lfloor i \\cdot H / H_{out} \\rfloor"}</Code>, <Code>{"h_{end} = \\lceil (i+1) \\cdot H / H_{out} \\rceil"}</Code>, and windows overlap when <Code>{"H_{out}"}</Code> does not divide <Code>H</Code> evenly.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
import torch
import torch.nn.functional as F

def max_pool2d(x, k=2, s=2):
    N, C, H, W = x.shape
    Hout, Wout = (H - k)//s + 1, (W - k)//s + 1
    out = np.zeros((N, C, Hout, Wout), dtype=x.dtype)
    for i in range(Hout):
        for j in range(Wout):
            out[:, :, i, j] = x[:, :, i*s:i*s+k, j*s:j*s+k].max(axis=(2, 3))
    return out

def avg_pool2d(x, k=2, s=2):
    N, C, H, W = x.shape
    Hout, Wout = (H - k)//s + 1, (W - k)//s + 1
    out = np.zeros((N, C, Hout, Wout), dtype=x.dtype)
    for i in range(Hout):
        for j in range(Wout):
            out[:, :, i, j] = x[:, :, i*s:i*s+k, j*s:j*s+k].mean(axis=(2, 3))
    return out

def global_avg_pool(x):
    return x.mean(axis=(2, 3), keepdims=True)

def adaptive_avg_pool(x, out_h, out_w):
    """Matches nn.AdaptiveAvgPool2d for arbitrary (out_h, out_w)."""
    N, C, H, W = x.shape
    out = np.zeros((N, C, out_h, out_w), dtype=x.dtype)
    for i in range(out_h):
        hs = (i * H) // out_h
        he = ((i + 1) * H + out_h - 1) // out_h
        for j in range(out_w):
            ws = (j * W) // out_w
            we = ((j + 1) * W + out_w - 1) // out_w
            out[:, :, i, j] = x[:, :, hs:he, ws:we].mean(axis=(2, 3))
    return out

np.random.seed(0)
x = np.random.randn(1, 2, 6, 6).astype(np.float32)
xt = torch.from_numpy(x)

checks = [
    ("max_pool2d(2,2)", max_pool2d(x), F.max_pool2d(xt, 2).numpy()),
    ("avg_pool2d(2,2)", avg_pool2d(x), F.avg_pool2d(xt, 2).numpy()),
    ("global_avg_pool", global_avg_pool(x), F.adaptive_avg_pool2d(xt, (1,1)).numpy()),
    ("adaptive(3x3)", adaptive_avg_pool(x, 3, 3), F.adaptive_avg_pool2d(xt, (3,3)).numpy()),
    ("adaptive(4x4)", adaptive_avg_pool(x, 4, 4), F.adaptive_avg_pool2d(xt, (4,4)).numpy()),
]
for name, ours, ref in checks:
    print(f"{name:<18} shape {ours.shape}  max-diff {np.abs(ours - ref).max():.2e}")

# Output:
# max_pool2d(2,2)    shape (1, 2, 3, 3)  max-diff 0.00e+00
# avg_pool2d(2,2)    shape (1, 2, 3, 3)  max-diff 0.00e+00
# global_avg_pool    shape (1, 2, 1, 1)  max-diff 2.98e-08
# adaptive(3x3)      shape (1, 2, 3, 3)  max-diff 0.00e+00
# adaptive(4x4)      shape (1, 2, 4, 4)  max-diff 0.00e+00`}
      </CodeBlock>

      <H3>4g. Depthwise separable convolution (preview)</H3>

      <Prose>
        Depthwise separable conv is the factorization at the heart of MobileNet, Xception, and every modern mobile CNN. It splits a standard conv into a <em>depthwise</em> per-channel spatial filter (groups=Cin) followed by a <em>pointwise</em> 1×1 conv that mixes channels. The next topic covers this in depth; here we simply verify the grouped-conv implementation matches a manually looped per-channel conv, and count the compute savings.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn, torch.nn.functional as F

Cin, Cout, H, W, k = 64, 128, 56, 56, 3

conv_std = nn.Conv2d(Cin, Cout, k, padding=1, bias=False)
dw = nn.Conv2d(Cin, Cin, k, padding=1, groups=Cin, bias=False)
pw = nn.Conv2d(Cin, Cout, 1, bias=False)

def n_params(m): return sum(p.numel() for p in m.parameters())
def conv_flops(Cin, Cout, H, W, k, g=1):
    return 2 * H * W * k * k * Cin * Cout // g

p_std = n_params(conv_std)
p_ds  = n_params(dw) + n_params(pw)
f_std = conv_flops(Cin, Cout, H, W, k)
f_ds  = conv_flops(Cin, Cin, H, W, k, g=Cin) + conv_flops(Cin, Cout, H, W, 1)

print(f"Standard 3x3 conv  : params = {p_std:>7}   FLOPs = {f_std/1e6:>7.2f} M")
print(f"Depthwise separable: params = {p_ds:>7}   FLOPs = {f_ds/1e6:>7.2f} M")
print(f"Reduction          : params x{p_std/p_ds:.2f}   FLOPs x{f_std/f_ds:.2f}")

# Grouped-conv equivalence check: groups=Cin is exactly per-channel conv
x = torch.randn(2, Cin, H, W)
y_dw = dw(x)
manual = torch.zeros_like(y_dw)
for c in range(Cin):
    manual[:, c:c+1] = F.conv2d(x[:, c:c+1], dw.weight[c:c+1], padding=1)
print(f"\\nGrouped vs per-channel loop: max abs diff = {(y_dw - manual).abs().max().item():.2e}")

# Output:
# Standard 3x3 conv  : params =   73728   FLOPs =  462.42 M
# Depthwise separable: params =    8768   FLOPs =   54.99 M
# Reduction          : params x8.41   FLOPs x8.41
#
# Grouped vs per-channel loop: max abs diff = 0.00e+00`}
      </CodeBlock>

      <Prose>
        At <Code>{"C_{in} = 64, C_{out} = 128, k = 3"}</Code> the depthwise separable version is 8.4× cheaper in both parameters and FLOPs. The savings grow with <Code>C</Code>: for <Code>{"C_{in} = C_{out} = 512"}</Code> and <Code>k = 3</Code>, the ratio is <Code>{"\\frac{k^2 \\cdot C_{in}}{1 + k^2 / C_{out}} \\approx k^2 = 9"}</Code>× — which is the quoted MobileNet number.
      </Prose>

      <H3>4h. Three-layer CNN forward pass with shape trace</H3>

      <CodeBlock language="python">
{`import torch, torch.nn as nn

torch.manual_seed(0)
x = torch.randn(1, 3, 32, 32)

layers = [
    ('conv1', nn.Conv2d(3, 16, 3, padding=1)),
    ('bn1',   nn.BatchNorm2d(16)),
    ('relu1', nn.ReLU()),
    ('pool1', nn.MaxPool2d(2)),              # 32 -> 16
    ('conv2', nn.Conv2d(16, 32, 3, padding=1)),
    ('bn2',   nn.BatchNorm2d(32)),
    ('relu2', nn.ReLU()),
    ('pool2', nn.MaxPool2d(2)),              # 16 -> 8
    ('conv3', nn.Conv2d(32, 64, 3, padding=1)),
    ('bn3',   nn.BatchNorm2d(64)),
    ('relu3', nn.ReLU()),
    ('gap',   nn.AdaptiveAvgPool2d((1, 1))), # 8 -> 1 (classifier head)
    ('flat',  nn.Flatten()),
    ('fc',    nn.Linear(64, 10)),
]

h = x
print(f"{'layer':<8} {'shape':<22} {'params':>8}")
print(f"{'input':<8} {str(tuple(h.shape)):<22} {0:>8}")
for name, m in layers:
    h = m(h)
    p = sum(p.numel() for p in m.parameters())
    print(f"{name:<8} {str(tuple(h.shape)):<22} {p:>8}")
total = sum(sum(p.numel() for p in m.parameters()) for _, m in layers)
print(f"\\ntotal params: {total:,}")

# Output:
# layer    shape                   params
# input    (1, 3, 32, 32)               0
# conv1    (1, 16, 32, 32)            448
# bn1      (1, 16, 32, 32)             32
# relu1    (1, 16, 32, 32)              0
# pool1    (1, 16, 16, 16)              0
# conv2    (1, 32, 16, 16)           4640
# bn2      (1, 32, 16, 16)             64
# relu2    (1, 32, 16, 16)              0
# pool2    (1, 32, 8, 8)                0
# conv3    (1, 64, 8, 8)            18496
# bn3      (1, 64, 8, 8)              128
# relu3    (1, 64, 8, 8)                0
# gap      (1, 64, 1, 1)                0
# flat     (1, 64)                      0
# fc       (1, 10)                    650
#
# total params: 24,458`}
      </CodeBlock>

      <Prose>
        A 24k-parameter CNN that would not fit a single FC layer on even a 32×32 image. Every conv adds parameters only proportional to <Code>{"k^2 \\cdot C_{in} \\cdot C_{out}"}</Code>; every pool halves spatial resolution for free; the final GAP plus linear replaces what would have been an enormous FC classifier. This is the template.
      </Prose>

      <H3>4i. VGG-16 FLOPs and parameter breakdown</H3>

      <CodeBlock language="python">
{`stack = [
    ('conv1_1', 3,   64,  224, 224, 3),
    ('conv1_2', 64,  64,  224, 224, 3),
    ('conv2_1', 64,  128, 112, 112, 3),
    ('conv2_2', 128, 128, 112, 112, 3),
    ('conv3_1', 128, 256, 56,  56,  3),
    ('conv3_2', 256, 256, 56,  56,  3),
    ('conv3_3', 256, 256, 56,  56,  3),
    ('conv4_1', 256, 512, 28,  28,  3),
    ('conv4_2', 512, 512, 28,  28,  3),
    ('conv4_3', 512, 512, 28,  28,  3),
    ('conv5_1', 512, 512, 14,  14,  3),
    ('conv5_2', 512, 512, 14,  14,  3),
    ('conv5_3', 512, 512, 14,  14,  3),
]
tot_p, tot_f = 0, 0
print(f"{'layer':<9}{'Cin':>4}{'Cout':>5}{'HxW':>10}{'params':>12}{'MFLOPs':>12}")
for name, Cin, Cout, H, W, k in stack:
    p = k * k * Cin * Cout
    f = 2 * H * W * k * k * Cin * Cout
    tot_p += p; tot_f += f
    print(f"{name:<9}{Cin:>4}{Cout:>5}{H}x{W:<6}{p:>12,}{f/1e6:>12.1f}")
print("-" * 52)
print(f"{'TOTAL':<9}{'':>4}{'':>5}{'':>10}{tot_p:>12,}{tot_f/1e6:>12.1f}")
fc = 512*7*7*4096 + 4096*4096 + 4096*1000
print(f"\\nFC classifier params (4096-4096-1000): {fc:,}")

# Output:
# layer     Cin Cout       HxW       params      MFLOPs
# conv1_1     3   64 224x224          1,728       173.4
# conv1_2    64   64 224x224         36,864      3699.4
# conv2_1    64  128 112x112         73,728      1849.7
# conv2_2   128  128 112x112        147,456      3699.4
# conv3_1   128  256 56x56          294,912      1849.7
# conv3_2   256  256 56x56          589,824      3699.4
# conv3_3   256  256 56x56          589,824      3699.4
# conv4_1   256  512 28x28        1,179,648      1849.7
# conv4_2   512  512 28x28        2,359,296      3699.4
# conv4_3   512  512 28x28        2,359,296      3699.4
# conv5_1   512  512 14x14        2,359,296       924.8
# conv5_2   512  512 14x14        2,359,296       924.8
# conv5_3   512  512 14x14        2,359,296       924.8
# ----------------------------------------------------
# TOTAL                           14,710,464     30693.3
#
# FC classifier params (4096-4096-1000): 123,633,664`}
      </CodeBlock>

      <Prose>
        VGG-16's convolutional trunk has 14.7 million parameters but costs 30.7 GFLOPs per forward pass. The FC head adds another 123.6 million parameters — 89% of the total VGG parameters — but costs only 0.12 GFLOPs. Conv is <em>cheap in parameters but expensive in compute</em>; FC is <em>expensive in parameters but cheap in compute</em>. This asymmetry is why GoogLeNet and ResNet replaced the FC head with global average pooling: almost no cost in accuracy, massive cost reduction in parameters.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5.1 torch.nn.Conv2d — the workhorse</H3>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn

# Full signature
conv = nn.Conv2d(
    in_channels=64,
    out_channels=128,
    kernel_size=3,       # int or (kH, kW)
    stride=1,            # int or (sH, sW)
    padding=1,           # int, tuple, or 'same'/'valid'
    dilation=1,          # int or (dH, dW)
    groups=1,            # 1 = dense; in_channels = depthwise
    bias=False,          # usually False when followed by BatchNorm
    padding_mode='zeros' # 'reflect', 'replicate', 'circular' also supported
)

# Common recipes
standard_3x3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
pointwise_1x1 = nn.Conv2d(256, 64, 1, bias=False)           # channel projection
depthwise_3x3 = nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False)
strided_downsample = nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)
dilated_3x3 = nn.Conv2d(64, 64, 3, padding=2, dilation=2, bias=False)

x = torch.randn(1, 64, 56, 56)
for name, m in [('3x3', standard_3x3), ('1x1', pointwise_1x1),
                ('dw3x3', depthwise_3x3), ('stride2', strided_downsample),
                ('dil2', dilated_3x3)]:
    print(f"{name:<8} in {tuple(x.shape)} -> out {tuple(m(x).shape)}  "
          f"params {sum(p.numel() for p in m.parameters()):,}")`}
      </CodeBlock>

      <H3>5.2 Pooling modules</H3>

      <CodeBlock language="python">
{`import torch.nn as nn

# Standard 2x2 max/avg pool used between conv blocks in VGG/ResNet-style nets
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

# Global Average Pooling for the classifier head.
# AdaptiveAvgPool2d((1,1)) works for any input spatial size -> always (N, C, 1, 1).
# This is what replaced the FC classifier in ResNet / GoogLeNet / ConvNeXt.
gap = nn.AdaptiveAvgPool2d((1, 1))

# Adaptive pool to fixed size (useful when input image sizes vary)
adap = nn.AdaptiveAvgPool2d((7, 7))   # as in torchvision.models.resnet's avgpool

# Modern classifier head template
classifier_head = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),  # (N, C, H, W) -> (N, C, 1, 1)
    nn.Flatten(),                   # (N, C, 1, 1) -> (N, C)
    nn.Linear(512, 1000),           # ImageNet logits
)

# Fractional / frac max pool and LP pool also exist but are rarely used in 2026.
print("maxpool / avgpool are stateless — no learnable parameters")`}
      </CodeBlock>

      <H3>5.3 Functional API, F.conv2d, and torch.nn.Unfold</H3>

      <Prose>
        When implementing custom layers, the stateless <Code>F.conv2d</Code> and <Code>torch.nn.Unfold</Code> are indispensable. <Code>Unfold</Code> is exactly the im2col operation — it turns any NCHW tensor into its unfolded patch matrix so you can run a custom matmul, apply per-patch operations, or implement attention-like conv variants.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn, torch.nn.functional as F

x = torch.randn(2, 3, 8, 8)
w = torch.randn(5, 3, 3, 3)

# Option A: functional conv
y = F.conv2d(x, w, padding=1)   # (2, 5, 8, 8)

# Option B: im2col via Unfold + explicit matmul
unfold = nn.Unfold(kernel_size=3, padding=1)
cols = unfold(x)                # (N, Cin*k*k, Hout*Wout) = (2, 27, 64)
w_mat = w.reshape(5, -1)        # (Cout, Cin*k*k) = (5, 27)
y2 = (w_mat @ cols).reshape(2, 5, 8, 8)

print(f"F.conv2d shape : {y.shape}")
print(f"unfold+matmul  : {y2.shape}")
print(f"max diff       : {(y - y2).abs().max().item():.2e}")

# Fold is the inverse (column-to-image) — used for transposed conv implementations
fold = nn.Fold(output_size=(8, 8), kernel_size=3, padding=1)
# Note: fold ACCUMULATES overlapping patches; divide by overlap count to average.`}
      </CodeBlock>

      <H3>5.4 Transposed convolution for upsampling</H3>

      <Prose>
        Transposed convolution (also "deconvolution", though that name is misleading) is the operation used to upsample feature maps — in segmentation decoders, GAN generators, and image-to-image models. It reverses the spatial reduction of a strided conv by inserting zeros between input values and running a standard conv. The output size formula is:
      </Prose>

      <MathBlock>
        {"n_{out} = (n_{in} - 1) \\cdot s - 2p + d \\cdot (k - 1) + \\text{output\\_padding} + 1"}
      </MathBlock>

      <CodeBlock language="python">
{`import torch, torch.nn as nn

# Common upsample recipe: 4x4 kernel, stride 2, pad 1 -> exactly doubles HxW
upsample = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                               kernel_size=4, stride=2, padding=1, bias=False)

x = torch.randn(1, 256, 14, 14)
y = upsample(x)
print(f"upsample: {tuple(x.shape)} -> {tuple(y.shape)}  "
      f"params {sum(p.numel() for p in upsample.parameters()):,}")
# upsample: (1, 256, 14, 14) -> (1, 128, 28, 28)

# Caution: stride=2 + kernel=3 produces CHECKERBOARD artifacts (Odena 2016).
# Modern alternative: bilinear upsample + 3x3 conv
upsample_safe = nn.Sequential(
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
    nn.Conv2d(256, 128, 3, padding=1, bias=False),
)
# No checkerboard, often produces smoother outputs for image generation tasks.`}
      </CodeBlock>

      <H3>5.5 cuDNN autotuning</H3>

      <Prose>
        cuDNN contains several algorithms for each conv configuration (IMPLICIT_GEMM, GEMM, WINOGRAD, FFT, FFT_TILING). On every new input shape, it can either pick an algorithm via heuristics or benchmark all of them and cache the fastest. For production training with fixed batch shapes, enabling benchmark mode can give 10–30% speedup after a short warmup phase.
      </Prose>

      <CodeBlock language="python">
{`import torch

# Enable cuDNN algorithm autotuning (benchmarks all algos on first forward,
# then caches the winner). Use when input shapes are STABLE across batches.
torch.backends.cudnn.benchmark = True

# Disable when input shapes vary (variable sequence length, variable image size)
# — otherwise the benchmark runs every time and dominates wall-clock.
# torch.backends.cudnn.benchmark = False

# Deterministic mode disables benchmark + forces deterministic algorithms.
# Slower but reproducible — required by some compliance workflows.
# torch.backends.cudnn.deterministic = True

# Query the chosen algorithm (available in PyTorch >= 2.1):
# torch.backends.cudnn.allow_tf32 = True   # Ampere+ accelerates FP32 conv via TF32`}
      </CodeBlock>

      <H3>5.6 Grouped conv and depthwise separable blocks</H3>

      <CodeBlock language="python">
{`import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """MobileNet-style block. Standard 3x3 conv replaced by DW3x3 + PW1x1."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # Depthwise: spatial filter, one per input channel
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                            groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        # Pointwise: channel mixer (1x1 conv = linear combination per pixel)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.bn1(self.dw(x)).relu()
        x = self.bn2(self.pw(x)).relu()
        return x

# At Cin=64, Cout=128, 3x3: 8.4x fewer params / FLOPs than standard 3x3 conv.
# Used in MobileNet-V1, V2, V3; Xception; EfficientNet; ConvNeXt (with kernel 7).`}
      </CodeBlock>

      <H3>5.7 Fused conv-bn-relu inference</H3>

      <Prose>
        At inference time, BatchNorm becomes a per-channel affine transform <Code>{"y = \\gamma \\cdot (x - \\mu) / \\sigma + \\beta"}</Code> with frozen statistics. This can be mathematically folded into the preceding conv's weights and bias — eliminating a kernel launch, saving memory, and on modern GPUs the fused conv-bn-relu often runs 1.5–2× faster than the three-operation sequence.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn

# torch.ao.quantization provides fuse_modules for this
from torch.ao.quantization import fuse_modules

class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

m = ConvBlock().eval()
# Fuse in place: Conv2d + BN + ReLU -> ConvBNReLU2d (single kernel at inference)
m_fused = fuse_modules(m, ['conv', 'bn', 'relu'])
# m_fused.conv now has merged weights; m_fused.bn and m_fused.relu are Identity.

# Alternative for deployment: torch.compile with inductor backend fuses these
# automatically at graph capture time.
# m_compiled = torch.compile(m, mode='reduce-overhead')`}
      </CodeBlock>

      <Callout accent="gold">
        Production defaults for convolutional layers in 2026: (1) <Code>bias=False</Code> on any conv followed by BN; (2) <Code>padding=kernel_size//2</Code> for same-resolution convs; (3) <Code>torch.backends.cudnn.benchmark=True</Code> for fixed-shape training; (4) fuse conv-bn-relu for inference; (5) use <Code>AdaptiveAvgPool2d((1,1))</Code> instead of a giant FC classifier head.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. 3×3 kernel sliding over a 5×5 input — output grid</H3>

      <Prose>
        For a 5×5 input and a 3×3 kernel with stride 1 and no padding, the output is 3×3. Each output cell is a weighted sum over a 3×3 window of the input, with the window translating one pixel per output column and one pixel per output row. The heatmap below visualizes the output: the value in cell (i, j) is the convolution response when the kernel is centered at input position (i+1, j+1). The input is a simple diagonal ramp and the kernel is a centered 3×3 averaging filter; the response pattern shows how the local average evolves across the 3×3 output grid.
      </Prose>

      <Heatmap
        label="3x3 conv output over a 5x5 diagonal-ramp input (k=3, s=1, p=0)"
        rowLabels={["out row 0", "out row 1", "out row 2"]}
        colLabels={["col 0", "col 1", "col 2"]}
        matrix={[
          [2.22, 3.33, 4.44],
          [3.33, 4.44, 5.56],
          [4.44, 5.56, 6.67],
        ]}
        colorScale="gold"
      />

      <Prose>
        The output is itself a diagonal ramp, preserved but spatially contracted from 5×5 to 3×3 — the "valid" convolution shrinks by <Code>k − 1 = 2</Code> pixels in each dimension. With <Code>padding = 1</Code>, the output would again be 5×5. The value at output (0, 0) = 2.22 is the average of input pixels (0,0)–(2,2); at output (2, 2) = 6.67 is the average of input pixels (2,2)–(4,4).
      </Prose>

      <H3>6b. Theoretical receptive field grows with depth (VGG-16)</H3>

      <Prose>
        The plot below traces the receptive field through VGG-16's 13 convolutional layers and 5 pooling layers. The RF grows linearly within each pool stage (each conv adds 2 pixels × current jump) and jumps sharply at each pool (doubles the jump, so subsequent convs contribute 2× as much). By <Code>pool5</Code> the RF is 212 pixels — essentially the whole 224×224 image.
      </Prose>

      <Plot
        label="VGG-16 theoretical receptive field by layer"
        xLabel="layer index"
        yLabel="receptive field (input pixels)"
        series={[
          {
            name: "theoretical RF",
            color: colors.gold,
            points: [
              [1, 3], [2, 5], [3, 6],
              [4, 10], [5, 14], [6, 16],
              [7, 24], [8, 32], [9, 40], [10, 44],
              [11, 60], [12, 76], [13, 92], [14, 100],
              [15, 132], [16, 164], [17, 196], [18, 212],
            ],
          },
        ]}
      />

      <Prose>
        The shape is characteristic: flat-then-jump, flat-then-jump. Every pool2 stride-2 doubles the current jump and causes the next conv stage to contribute more per layer than the previous stage did. This is why networks gain RF fastest right after a stride-2 downsample. It is also why extra conv layers at the final resolution (after pool5) would add relatively little new context — at jump 32, each further 3×3 conv adds 64 pixels of RF per layer but the image is only 224 pixels wide, so you saturate almost immediately.
      </Prose>

      <H3>6c. Theoretical vs effective receptive field — the ERF gap</H3>

      <Prose>
        Using the Luo 2016 measurement from Section 4e, we overlay theoretical RF (linear in depth) against empirically measured ERF (1% cutoff) for stacks of 3×3 convs at depths 5, 10, 15, 20. The gap widens with depth: by 20 layers the ERF is only half the theoretical RF and the gap continues to grow as <Code>{"\\sqrt{N}"}</Code> vs <Code>N</Code>.
      </Prose>

      <Plot
        label="theoretical RF vs effective RF (measured) for stacks of 3x3 convs"
        xLabel="number of 3x3 conv layers"
        yLabel="receptive field (pixels)"
        series={[
          {
            name: "theoretical RF = 1 + 2N",
            color: colors.gold,
            points: [[5, 11], [10, 21], [15, 31], [20, 41]],
          },
          {
            name: "effective RF (1% cutoff)",
            color: colors.green,
            points: [[5, 8], [10, 16], [15, 18], [20, 21]],
          },
        ]}
      />

      <Prose>
        The green line (ERF) grows sublinearly; the gold line (theoretical RF) is strictly linear. In a real network — with strided downsampling, learned weights, and ReLU nonlinearities — the effect is even more pronounced. Luo et al. measured a trained ResNet-34 on CIFAR and found an effective RF of roughly 32×32 pixels out of a theoretical 896×896 (model trained on 224 input, so the theoretical RF wraps around many times).
      </Prose>

      <H3>6d. AlexNet conv1 — hand-illustrative 11×11 kernels</H3>

      <Prose>
        AlexNet's first layer has 96 filters of size 11×11×3, trained on ImageNet. The learned filters famously split into two groups: Gabor-like oriented edge detectors (appearing in both GPU halves of the original 2-GPU training) and color blobs (concentrated on one GPU). The heatmap below is an illustrative reconstruction of one 11×11 oriented-edge filter's luminance channel — the classical pattern that also matches V1 simple-cell receptive fields measured by Hubel & Wiesel.
      </Prose>

      <Heatmap
        label="illustrative AlexNet conv1 filter — oriented edge (luminance channel)"
        rowLabels={["r0","r1","r2","r3","r4","r5","r6","r7","r8","r9","r10"]}
        colLabels={["c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","c10"]}
        matrix={[
          [-0.3,-0.3,-0.2,-0.1, 0.0, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
          [-0.4,-0.4,-0.3,-0.2, 0.0, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2],
          [-0.5,-0.5,-0.4,-0.2, 0.0, 0.2, 0.4, 0.5, 0.5, 0.4, 0.2],
          [-0.6,-0.6,-0.5,-0.3, 0.0, 0.3, 0.5, 0.6, 0.6, 0.5, 0.3],
          [-0.7,-0.7,-0.6,-0.3, 0.0, 0.3, 0.6, 0.7, 0.7, 0.6, 0.3],
          [-0.7,-0.7,-0.6,-0.3, 0.0, 0.3, 0.6, 0.7, 0.7, 0.6, 0.3],
          [-0.7,-0.7,-0.6,-0.3, 0.0, 0.3, 0.6, 0.7, 0.7, 0.6, 0.3],
          [-0.6,-0.6,-0.5,-0.3, 0.0, 0.3, 0.5, 0.6, 0.6, 0.5, 0.3],
          [-0.5,-0.5,-0.4,-0.2, 0.0, 0.2, 0.4, 0.5, 0.5, 0.4, 0.2],
          [-0.4,-0.4,-0.3,-0.2, 0.0, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2],
          [-0.3,-0.3,-0.2,-0.1, 0.0, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
        ]}
        colorScale="gold"
      />

      <Prose>
        The zero-crossing runs vertically down column 4 — negative weights on the left, positive on the right — which is exactly the response of a vertical edge detector. When convolved with an image containing a vertical luminance discontinuity, this filter lights up. The fact that AlexNet's 96 filters spontaneously arranged themselves into this classical Gabor / color-blob dictionary was a major validation that convolutional features match the primary visual cortex — the Hubel & Wiesel findings, learned from scratch via SGD.
      </Prose>

      <H3>6e. Three-layer CNN forward pass — shapes at each step</H3>

      <StepTrace
        label="forward pass through 3-layer CNN on a 32x32x3 input"
        steps={[
          {
            label: "Step 1 — Input arrives",
            render: () => (
              <Prose>
                {"Input tensor of shape (1, 3, 32, 32): one RGB image, 32 x 32 pixels. Pixel range is typically [0, 1] after division by 255, or standardized to mean 0 std 1 per channel after normalization. Channel dimension is second by PyTorch convention (NCHW). NHWC layout is faster on some hardware (TensorCore requires it internally) but PyTorch surfaces NCHW to users."}
              </Prose>
            ),
          },
          {
            label: "Step 2 — conv1 (3 -> 16, k=3, p=1)",
            render: () => (
              <Prose>
                {"Conv2d(3, 16, kernel=3, padding=1) produces (1, 16, 32, 32). Spatial dimensions preserved because padding = kernel_size // 2. Parameter count = 3 * 3 * 3 * 16 + 16 = 448 (9 kernel weights times 3 input channels times 16 output channels, plus 16 biases). Each output channel is a learned 3x3x3 filter responding to a different pattern in the input."}
              </Prose>
            ),
          },
          {
            label: "Step 3 — bn1 + relu1",
            render: () => (
              <Prose>
                {"BatchNorm2d(16) normalizes each channel to zero mean / unit variance across the batch plus the spatial axes, then applies a learnable per-channel affine (gamma, beta). Shape unchanged: (1, 16, 32, 32). ReLU clamps negatives to zero, shape still (1, 16, 32, 32). The BN stats require batch_size > 1 to be stable during training — at batch 1, consider GroupNorm or LayerNorm."}
              </Prose>
            ),
          },
          {
            label: "Step 4 — pool1 (MaxPool2d(2))",
            render: () => (
              <Prose>
                {"2x2 max pool with stride 2 halves spatial resolution: (1, 16, 32, 32) -> (1, 16, 16, 16). For each 2x2 non-overlapping block in each channel, output the maximum. This gains partial translation invariance (small shifts within the window map to the same output) and reduces compute for the next conv layer by 4x. No learnable parameters."}
              </Prose>
            ),
          },
          {
            label: "Step 5 — conv2 (16 -> 32, k=3, p=1)",
            render: () => (
              <Prose>
                {"Conv2d(16, 32, kernel=3, padding=1) produces (1, 32, 16, 16). Parameter count = 3 * 3 * 16 * 32 + 32 = 4640. At this stage the kernel sees a 3x3 window of the 16x16 feature map, which corresponds to pixels 3x2 = 6 input pixels wide (RF growth: previous RF was 3, pool doubled jump, so this conv contributes 2 * 2 = 4 extra input pixels -> total RF = 7 input pixels)."}
              </Prose>
            ),
          },
          {
            label: "Step 6 — bn2 + relu2 + pool2",
            render: () => (
              <Prose>
                {"BatchNorm2d(32) + ReLU + MaxPool2d(2) with stride 2. Output shape: (1, 32, 8, 8). After this second pool the jump is 4 and the RF has grown to about 15 input pixels. The channel count doubled from 16 to 32 — a common pattern: as spatial resolution halves, channels double to preserve total representational capacity."}
              </Prose>
            ),
          },
          {
            label: "Step 7 — conv3 (32 -> 64, k=3, p=1) + bn3 + relu3",
            render: () => (
              <Prose>
                {"Conv2d(32, 64, 3, padding=1) produces (1, 64, 8, 8). The biggest weight layer so far at 18,496 parameters. RF now reaches about 23 input pixels — larger than many CIFAR objects. The network has moved from local edge detectors at conv1 to object-scale features at conv3."}
              </Prose>
            ),
          },
          {
            label: "Step 8 — GAP (AdaptiveAvgPool2d((1,1)))",
            render: () => (
              <Prose>
                {"Global Average Pooling collapses the 8x8 spatial dimensions to 1x1: (1, 64, 8, 8) -> (1, 64, 1, 1). Each channel becomes its mean over the final feature map. This single operation replaces what would have been 64 * 8 * 8 * 10 = 40,960 FC weights if flattened directly. GAP is also translation-invariant — shifts in the input do not change the output — and significantly reduces overfitting."}
              </Prose>
            ),
          },
          {
            label: "Step 9 — Flatten + Linear to 10 logits",
            render: () => (
              <Prose>
                {"Flatten (1, 64, 1, 1) -> (1, 64). Linear(64, 10) produces 10 class logits for CIFAR-10, adding 64*10 + 10 = 650 parameters. Total network: ~24,000 parameters, competitive on CIFAR-10 with simple training. This structure — conv trunk, GAP, linear classifier — is the template for ResNet, MobileNet, EfficientNet, and ConvNeXt."}
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>7.1 Kernel size</H3>

      <Prose>
        <strong>3×3 is the default for almost every situation.</strong> VGG established this in 2014 and nothing since has dislodged it. Three reasons: (1) stacking two 3×3 convs gives a 5×5 RF with 18 params per channel-pair vs. 25 for a single 5×5, (2) every extra 3×3 adds a non-linearity, so two 3×3s are strictly more expressive than one 5×5, (3) cuDNN's Winograd algorithm for 3×3 runs at ~2× the FLOPs of a matmul, which is the best FLOP/throughput ratio of any conv configuration.
      </Prose>

      <Prose>
        <strong>1×1 convs for channel projection.</strong> A 1×1 conv is a per-pixel linear map from <Code>{"C_{in}"}</Code> to <Code>{"C_{out}"}</Code> channels. Use it to (a) reduce channel count before an expensive 3×3 (bottleneck blocks in ResNet-50), (b) expand channels after a depthwise conv (MobileNet), or (c) cheaply add nonlinearity with a bn+relu after each 1×1 at almost no FLOP cost.
      </Prose>

      <Prose>
        <strong>5×5 and 7×7 early.</strong> AlexNet's first layer is 11×11; GoogLeNet uses 7×7 at stem. Large early-layer kernels aggressively downsample (stride 2 or 4) and extract large-scale features cheaply — at the input, <Code>{"C_{in} = 3"}</Code> so even an 11×11 costs little. ConvNeXt revived this pattern in 2022 with 7×7 depthwise convs at the stem.
      </Prose>

      <Prose>
        <strong>Very large kernels (31×31 and beyond).</strong> RepLKNet (2022) and SLaK (2022) showed that for a fixed FLOP budget, a 31×31 depthwise conv can outperform a stack of 3×3 convs for dense prediction — because the ERF increase matters more than adding more non-linearities. Niche but growing.
      </Prose>

      <H3>7.2 Pooling strategy</H3>

      <Prose>
        <strong>Max pool vs average pool vs strided conv.</strong> Max pool is sharp: it keeps the strongest response in each window and discards the rest. Average pool is smooth: it aggregates all responses equally. Strided conv is learned: the downsampling pattern is optimized jointly with the filter. Modern networks (ResNet, ConvNeXt, vision Transformers' patchify stem) almost always use strided convolution for mid-network downsampling rather than max pool — the learned filter usually wins.
      </Prose>

      <Prose>
        <strong>Global average pooling for classifier heads.</strong> GAP replaces the giant FC classifier of AlexNet/VGG and is now universal. The feature map produced by the final conv is collapsed spatially to one value per channel, then a single linear layer produces class logits. Eliminates hundreds of millions of parameters with no accuracy loss — ResNet-50 has 25M parameters largely because of GAP.
      </Prose>

      <Prose>
        <strong>Max pool for feature-selection tasks.</strong> Where you want the strongest activation to win (detection, anomaly detection, certain attention pooling schemes), max pool is appropriate. For general feature aggregation, average pool is usually better because its gradient distributes over all inputs rather than routing only to the single winner.
      </Prose>

      <H3>7.3 How to grow the receptive field</H3>

      <Prose>
        <strong>Dilated convolution for dense prediction.</strong> When you need large RF but cannot downsample (semantic segmentation, depth estimation), dilation is the primary tool. Cascaded dilated convs (DeepLab, PSPNet's atrous spatial pyramid pooling) stack <Code>d = 1, 2, 4, 8</Code> to cover multiple scales without spatial reduction. Be careful: dilation creates gridding artifacts if multiple successive layers use the same dilation — vary the rates.
      </Prose>

      <Prose>
        <strong>Strided downsampling for image classification.</strong> When spatial fidelity is expendable (you only need one label per image), aggressive stride-2 downsampling is faster than dilation and produces larger effective RF for the same compute. ResNet-50 downsamples 5 times (4 in the trunk plus the initial stem), reaching 32× downsample and RF ≈ input size by the end.
      </Prose>

      <Prose>
        <strong>Attention for truly global context.</strong> The Transformer makes every output position attend to every input position — effective RF is the full input by construction. At high resolution this is <Code>{"O(N^4)"}</Code> and infeasible, which is why hybrid CNN-ViT models (Swin, MaxViT, CoAtNet) use local attention windows plus downsampling.
      </Prose>

      <H3>7.4 Efficient variants</H3>

      <Prose>
        <strong>Depthwise separable for mobile.</strong> When the target is a phone or edge device, depthwise separable 3×3 + pointwise 1×1 gives ~9× fewer FLOPs than dense 3×3 at similar accuracy. Every recent mobile architecture (MobileNet, EfficientNet, MobileViT) uses this.
      </Prose>

      <Prose>
        <strong>Grouped conv for throughput.</strong> Grouping inputs into <Code>g</Code> parallel subproblems divides FLOPs by <Code>g</Code>. ResNeXt-50-32×4d uses cardinality 32 and improves ImageNet accuracy at the same FLOP budget. Regnet and EfficientNet use moderate grouping as a tunable axis.
      </Prose>

      <Prose>
        <strong>Matrix-decomposed kernels.</strong> Replacing a 5×5 with 1×5 + 5×1 saves parameters (10 instead of 25). Inception-V3 used this for 1×7 and 7×1 decompositions. Largely superseded by the 3×3 stacking approach but appears in attention approximations and efficient depthwise designs.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <H3>8.1 Conv FLOPs dominate training time</H3>

      <Prose>
        For ResNet-50, roughly 80–90% of forward-pass FLOPs are in the convolutional trunk; the remainder is BN, ReLU, and the final FC. For EfficientNet and MobileNet the ratio is even more conv-dominant because depthwise separable blocks replace the relatively small FC layers. On A100 and H100 GPUs, a 3×3 dense conv at typical vision-model shapes runs at 40–60% of peak FP32 throughput through cuDNN GEMM or Winograd. This is why almost every conv performance optimization — Winograd, FFT, Tensor Cores, mixed precision — targets the conv layer specifically.
      </Prose>

      <H3>8.2 Winograd for 3×3 convolutions</H3>

      <Prose>
        Winograd (Lavin & Gray 2016) reduces the number of multiplications in a conv from <Code>{"k^2"}</Code> to <Code>{"(k + r - 1)^2 / r^2"}</Code> for an <Code>r × r</Code> output tile. For 3×3 kernels with 2×2 output tiles (the F(2×2, 3×3) transform), the reduction is 4 multiplications per output pixel per channel pair instead of 9 — a 2.25× theoretical speedup. cuDNN's Winograd algorithm is the default for 3×3 stride-1 convs at input spatial sizes above ~28×28 and below ~512×512, where the transform's constant overhead is amortized. Larger kernels or larger tiles require more additions, so the optimal balance point is F(2×2, 3×3) or F(4×4, 3×3).
      </Prose>

      <H3>8.3 FFT for large kernels</H3>

      <Prose>
        For kernel sizes above ~15×15, FFT-based convolution (O(n² log n)) beats direct convolution (O(n² k²)). cuDNN exposes <Code>FFT</Code> and <Code>FFT_TILING</Code> algorithms for these cases. Large-kernel architectures like RepLKNet (31×31 depthwise) rely on FFT at inference. At the small kernel sizes used for most vision models (3×3 to 7×7), FFT's constant factors make it slower than Winograd or direct GEMM.
      </Prose>

      <H3>8.4 Tensor Cores and mixed precision</H3>

      <Prose>
        Since Volta (2017), NVIDIA GPUs have contained dedicated matrix-multiply units (Tensor Cores) that run FP16/BF16 matmul at 4–8× the throughput of FP32. Because cuDNN convolution reduces to matmul, Tensor Cores directly accelerate conv layers. Using <Code>torch.cuda.amp.autocast(dtype=torch.bfloat16)</Code> or training with BF16 end-to-end typically delivers 2× throughput on A100 and higher on H100 with negligible accuracy loss for vision models. Ampere added FP8 support via TransformerEngine — widely used for LLM training, less for vision CNN training where the numerical dynamic range is narrower.
      </Prose>

      <H3>8.5 Structured sparsity (Ampere 2:4)</H3>

      <Prose>
        NVIDIA A100 and later accelerate matmul by 2× when weight matrices follow a 2:4 sparsity pattern (exactly 2 of every 4 contiguous values are zero). The <Code>torch.sparse</Code> and <Code>apex.contrib</Code> paths can prune a trained conv's weights to 2:4 and re-fine-tune, preserving accuracy while doubling inference throughput. Most production LLM inference stacks now use 2:4 sparsity; vision CNN deployment is catching up.
      </Prose>

      <H3>8.6 Fused kernels</H3>

      <Prose>
        At inference, conv + BN + ReLU (and often + residual add) are fused into a single CUDA kernel — one memory read, one kernel launch, one memory write — instead of three or four separate kernels. This removes ~30–40% of memory traffic at small batch sizes where memory bandwidth rather than compute bounds throughput. <Code>torch.compile</Code> and TensorRT both perform this fusion automatically; <Code>torch.ao.quantization.fuse_modules</Code> does it manually.
      </Prose>

      <H3>8.7 Memory layout (channels-last)</H3>

      <Prose>
        PyTorch's default is NCHW but Tensor Cores prefer NHWC (channels-last) because it aligns with how the GEMM input matrix is laid out after im2col. Converting a model to <Code>memory_format=torch.channels_last</Code> at training time yields 10–30% additional speedup on H100 for common vision models. Every production training recipe for ResNet/EfficientNet/ConvNeXt uses channels-last.
      </Prose>

      <Prose>
        Together, these optimizations (Tensor Cores + BF16 + channels-last + fused conv-bn-relu) give a roughly 4–6× throughput improvement over naive FP32 NCHW on the same hardware. A ResNet-50 ImageNet training that took ~30 minutes per epoch on a 4×V100 system in 2017 now takes ~3 minutes on a single H100 in 2026, and a large fraction of that gap is these conv-specific optimizations.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Forgetting padding — silent shrinkage</H3>

      <Prose>
        A 3×3 conv with no padding shrinks the feature map by 2 pixels per layer. Stack 10 such convs and a 32×32 input becomes 12×12 — easily forgotten until the final pool or FC layer shape-errors. The fix is always <Code>padding = kernel_size // 2</Code> for odd kernels. For even kernels the padding is asymmetric (pad 1 on one side, 0 on the other) which is why essentially no architecture uses even kernels except for transposed conv upsampling.
      </Prose>

      <H3>9.2 Stride + pool double downsample</H3>

      <Prose>
        A common bug in handwritten architectures: a stride-2 conv immediately followed by a 2×2 stride-2 pool drops resolution by 4× in one block instead of 2×, ruining the spatial hierarchy. Pick one downsampling operation per stage — either the conv strides, or the pool strides, but not both. ResNet chose conv striding (cleaner RF growth, learnable); VGG chose pooling; mixing the two inside a single stage leads to aliasing and information loss.
      </Prose>

      <H3>9.3 Odd vs even kernels — asymmetric padding</H3>

      <Prose>
        Odd kernels (3, 5, 7) center cleanly: pad <Code>(k-1)/2</Code> on both sides, output has the same spatial center as the input. Even kernels (2, 4) require asymmetric padding because <Code>k/2</Code> is not a whole half-pixel. PyTorch's <Code>padding='same'</Code> handles this by padding 0 on left, 1 on right — but this shifts the spatial center by half a pixel per layer, which compounds and causes subtle misalignment between features at different depths. Use odd kernels unless you have a specific reason not to.
      </Prose>

      <H3>9.4 Groups must divide channels</H3>

      <Prose>
        <Code>nn.Conv2d(Cin, Cout, k, groups=g)</Code> requires <Code>g</Code> to divide both <Code>Cin</Code> and <Code>Cout</Code>. If you set <Code>groups=4</Code> on a <Code>Cin=10</Code> layer, PyTorch throws <Code>RuntimeError: in_channels must be divisible by groups</Code> at module construction. This error is caught early; the harder bug is when you forget the divisibility constraint while computing <Code>Cin</Code> dynamically from a prior layer, and the runtime error surfaces only during the first forward pass after hours of training setup.
      </Prose>

      <H3>9.5 Transposed conv checkerboard artifacts</H3>

      <Prose>
        In 2016, Augustus Odena, Vincent Dumoulin, and Chris Olah published "Deconvolution and Checkerboard Artifacts" on Distill. They showed that <Code>ConvTranspose2d</Code> with <Code>kernel_size = 3, stride = 2</Code> produces a periodic checkerboard pattern because adjacent output pixels receive contributions from different numbers of kernel taps (some from 2 taps, some from 1). The fix is either (a) kernel size divisible by stride (<Code>kernel_size=4, stride=2</Code> works cleanly), or (b) replace transposed conv with bilinear upsample + 3×3 conv. Every modern GAN and diffusion decoder uses the latter pattern.
      </Prose>

      <H3>9.6 Pooling is lossy</H3>

      <Prose>
        Pooling throws away information. A 2×2 max pool keeps only 1 of 4 input pixels' max value per channel; the remaining three are discarded. For classification this is usually fine — the discarded information was redundant — but for dense prediction (segmentation, super-resolution, keypoint detection) any pooling forces you to upsample later, and the upsampled feature map cannot recover the lost detail without a skip connection from the pre-pool feature map. This is why U-Net and FPN architectures carry skip connections across every downsample/upsample pair.
      </Prose>

      <H3>9.7 Dropout before pool</H3>

      <Prose>
        Applying dropout to a conv feature map, then max-pooling, is statistically broken. Max pool always selects the maximum, so if dropout randomly zeroed the would-be maximum, the pool silently picks a different value — which is not what dropout was supposed to do. The fix is either (a) dropout after pooling, (b) use SpatialDropout (channel-wise dropout that zeroes entire channels so the pool degrades gracefully), or (c) skip dropout in conv blocks entirely and rely on BatchNorm's implicit regularization, which is what ResNet and nearly all modern CNNs do.
      </Prose>

      <H3>9.8 Theoretical RF is not effective RF</H3>

      <Prose>
        Luo et al. 2016's main warning: just because your theoretical RF covers the whole image does not mean your network actually uses that context. Symptom: a 100-layer CNN on 1024×1024 inputs that somehow cannot learn object context spanning more than 200 pixels. Diagnosis: measure the ERF empirically (Section 4e). Remediation: dilation, large-kernel layers, or explicit long-range mechanisms (attention, non-local blocks). Stacking more small-kernel convs is not a path to large effective context.
      </Prose>

      <H3>9.9 BatchNorm statistics at inference</H3>

      <Prose>
        Not strictly a conv bug, but coupled: if you forget <Code>model.eval()</Code> before inference, BatchNorm will recompute statistics from the current batch (often batch size 1), produce wildly wrong outputs, and if gradients are still being tracked, leak memory until OOM. The symptom is inference that works on batches of 64 but catastrophically fails on batch 1. Always <Code>model.eval()</Code> and wrap inference in <Code>torch.no_grad()</Code> or use <Code>torch.inference_mode()</Code> (faster than <Code>no_grad</Code> in PyTorch 1.9+).
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The papers below are the foundational reading for convolution, pooling, and receptive field analysis. Hubel &amp; Wiesel for the neuroscience; Fukushima, LeCun, and Krizhevsky for the architecture lineage; Luo for the effective RF story; Dumoulin &amp; Visin for the formal arithmetic; Yu &amp; Koltun for dilated convolutions.
      </Prose>

      <Prose>
        <strong>Hubel, D.H. &amp; Wiesel, T.N. (1962).</strong> "Receptive fields, binocular interaction and functional architecture in the cat's visual cortex." <em>The Journal of Physiology</em>, 160(1), 106–154. The experimental paper identifying simple and complex cells in V1, which forms the neurophysiological inspiration for the CNN's alternation of convolution and pooling. Awarded the 1981 Nobel Prize in Physiology or Medicine (shared with Sperry).
      </Prose>

      <Prose>
        <strong>Fukushima, K. (1980).</strong> "Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position." <em>Biological Cybernetics</em>, 36(4), 193–202. The first hierarchical convolutional architecture, alternating S-layers (convolutional feature detectors) with C-layers (pooling / shift invariance). Trained with an unsupervised competitive rule rather than backprop, which arrived nine years later.
      </Prose>

      <Prose>
        <strong>LeCun, Y., Boser, B., Denker, J.S., Henderson, D., Howard, R.E., Hubbard, W. &amp; Jackel, L.D. (1989).</strong> "Backpropagation Applied to Handwritten Zip Code Recognition." <em>Neural Computation</em>, 1(4), 541–551. First end-to-end backprop-trained convolutional network on real-world data (USPS ZIP code digits). Introduced weight sharing and local receptive fields as engineering techniques rather than biological principles.
      </Prose>

      <Prose>
        <strong>LeCun, Y., Bottou, L., Bengio, Y. &amp; Haffner, P. (1998).</strong> "Gradient-Based Learning Applied to Document Recognition." <em>Proceedings of the IEEE</em>, 86(11), 2278–2324. The LeNet-5 paper. Also a comprehensive tutorial on gradient-based machine learning that fixed terminology across the field. Deployed in production to read ~60M checks/month.
      </Prose>

      <Prose>
        <strong>Krizhevsky, A., Sutskever, I. &amp; Hinton, G.E. (2012).</strong> "ImageNet Classification with Deep Convolutional Neural Networks." <em>NeurIPS 2012</em>. AlexNet. The paper that ended classical computer vision by winning ILSVRC 2012 with a 10.8-point lead over hand-crafted features. Introduced ReLU at scale, dropout on fully connected layers, GPU training, and local response normalization.
      </Prose>

      <Prose>
        <strong>Luo, W., Li, Y., Urtasun, R. &amp; Zemel, R. (2016).</strong> "Understanding the Effective Receptive Field in Deep Convolutional Neural Networks." <em>NeurIPS 2016</em>, arXiv:1701.04128. Shows that the effective RF of a CNN is much smaller than the theoretical RF, grows like <Code>{"\\sqrt{N}"}</Code> with depth instead of <Code>N</Code>, and has Gaussian spatial profile. Reshaped how dense-prediction architectures are designed.
      </Prose>

      <Prose>
        <strong>Dumoulin, V. &amp; Visin, F. (2016).</strong> "A guide to convolution arithmetic for deep learning." arXiv:1603.07285. The definitive reference for output size formulas, padding conventions, transposed convolutions, and dilated convolutions. Thirty-four pages of worked examples with animated figures in the companion GitHub repo (vdumoulin/conv_arithmetic).
      </Prose>

      <Prose>
        <strong>Yu, F. &amp; Koltun, V. (2015).</strong> "Multi-Scale Context Aggregation by Dilated Convolutions." arXiv:1511.07122, ICLR 2016. Introduced dilated (atrous) convolutions as a way to grow receptive field exponentially with depth without downsampling. The foundation of DeepLab, WaveNet's causal dilations, and every subsequent dense-prediction architecture that needs multi-scale context.
      </Prose>

      <Prose>
        <strong>Further reading worth the time in 2026:</strong> He et al. 2015 "Deep Residual Learning for Image Recognition" (ResNet — covered in the next topic); Simonyan &amp; Zisserman 2014 "Very Deep Convolutional Networks for Large-Scale Image Recognition" (VGG); Szegedy et al. 2014 "Going Deeper with Convolutions" (GoogLeNet / Inception); Howard et al. 2017 "MobileNets" (depthwise separable); Liu et al. 2022 "A ConvNet for the 2020s" (ConvNeXt — modernized CNN matching ViT); Ding et al. 2022 "Scaling Up Your Kernels to 31×31" (RepLKNet, the case for very large kernels). Odena et al. 2016 "Deconvolution and Checkerboard Artifacts" on Distill remains the clearest visual explanation of transposed-conv failure modes.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check</H2>

      <H3>Q1 — Output size with padding, stride, and dilation</H3>

      <Prose>
        An input feature map has spatial size 56×56. A convolution with kernel size 3, padding 2, stride 1, dilation 2 is applied. What is the output spatial size?
      </Prose>

      <Callout accent="gold">
        <strong>Answer.</strong> Effective kernel = <Code>{"d \\cdot (k - 1) + 1 = 2 \\cdot 2 + 1 = 5"}</Code>. Plugging into the formula: <Code>{"\\lfloor (56 + 4 - 5) / 1 \\rfloor + 1 = 55 + 1 = 56"}</Code>. The output is 56×56 — dilation 2 with padding 2 preserves spatial size for a 3×3 kernel, analogous to how dilation 1 with padding 1 does. This is the standard "same" configuration for dilated convs.
      </Callout>

      <H3>Q2 — Receptive field of 3 stacked 3×3 convs</H3>

      <Prose>
        Three 3×3 convolutions are stacked with stride 1 and appropriate padding. What is the theoretical receptive field of a neuron in the third layer's output? How many parameters are required (ignoring biases) if each conv has <Code>C</Code> input and output channels?
      </Prose>

      <Callout accent="gold">
        <strong>Answer.</strong> RF recursion: <Code>{"r_1 = 3, r_2 = r_1 + 2 = 5, r_3 = r_2 + 2 = 7"}</Code>. So the RF is 7×7 — the same as a single 7×7 conv. Parameters: 3 convs × <Code>{"3 \\cdot 3 \\cdot C \\cdot C = 9C^2"}</Code> each = <Code>{"27 C^2"}</Code>. A single 7×7 conv would be <Code>{"49 C^2"}</Code>. Stacking three 3×3s is 45% cheaper <em>and</em> adds 2 extra non-linearities. This is the VGG insight.
      </Callout>

      <H3>Q3 — Theoretical vs effective receptive field</H3>

      <Prose>
        You build a CNN with 50 3×3 conv layers, stride 1 throughout, and measure the empirical effective RF at the center of the output. The theoretical RF is 101 pixels. Would you expect the empirical ERF to be much smaller, roughly equal, or much larger? Why?
      </Prose>

      <Callout accent="gold">
        <strong>Answer.</strong> Much smaller. Luo et al. 2016 shows that the effective RF grows like <Code>{"\\sqrt{N}"}</Code> while the theoretical RF grows like <Code>N</Code>, so the ERF at depth 50 is on the order of <Code>{"\\sqrt{50} \\approx 7"}</Code> — small dozens of pixels, not 101. The effective RF has a Gaussian spatial profile: most input pixels the output could theoretically see contribute almost nothing. Practical consequence: stacking more small-kernel convs has diminishing returns for enlarging usable context; dilation or large kernels are more effective.
      </Callout>

      <H3>Q4 — Depthwise separable FLOP savings</H3>

      <Prose>
        A standard 3×3 convolution with <Code>{"C_{in} = 256, C_{out} = 256"}</Code> operates on a 28×28 feature map. What is its FLOP count? Compared with the depthwise-separable equivalent (depthwise 3×3 then pointwise 1×1), what is the FLOP saving?
      </Prose>

      <Callout accent="gold">
        <strong>Answer.</strong> Standard: <Code>{"2 \\cdot 28^2 \\cdot 9 \\cdot 256 \\cdot 256 = 2 \\cdot 784 \\cdot 9 \\cdot 65536 \\approx 924 \\text{ MFLOPs}"}</Code>. Depthwise: <Code>{"2 \\cdot 784 \\cdot 9 \\cdot 256 = 3.6 \\text{ MFLOPs}"}</Code>. Pointwise: <Code>{"2 \\cdot 784 \\cdot 1 \\cdot 256 \\cdot 256 \\approx 103 \\text{ MFLOPs}"}</Code>. Separable total ≈ 107 MFLOPs. Ratio ≈ 8.7×, converging toward <Code>{"k^2 = 9"}</Code> as channel count grows. This is the headline MobileNet number.
      </Callout>

      <H3>Q5 — Global average pooling vs flatten + FC</H3>

      <Prose>
        Your classification head takes a feature map of shape <Code>{"(B, 512, 7, 7)"}</Code> and produces 1000 class logits. Compare two options: (A) flatten to <Code>{"(B, 25088)"}</Code> then a single <Code>{"\\text{Linear}(25088, 1000)"}</Code>; (B) global average pool to <Code>{"(B, 512, 1, 1)"}</Code> then <Code>{"\\text{Linear}(512, 1000)"}</Code>. What are the parameter counts and when would you choose each?
      </Prose>

      <Callout accent="gold">
        <strong>Answer.</strong> Option A has <Code>{"25088 \\cdot 1000 + 1000 = 25{,}089{,}000"}</Code> parameters. Option B has <Code>{"512 \\cdot 1000 + 1000 = 513{,}000"}</Code> parameters — 49× fewer. Choose B for: image classification (translation-invariant task, enormous parameter savings, minimal accuracy loss — every modern CNN from 2015 onward uses it). Choose A only when spatial position matters to the output (e.g., predicting where an object is rather than what it is) and you have a small spatial grid; but in that case, you would typically use 1×1 conv to logits rather than flatten + FC, which preserves spatial structure even better.
      </Callout>

    </div>
  ),
};

export default convolutionPoolingRFContent;
