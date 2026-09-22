// Full prepared revision-3 manuscript; static JSX, no browser Markdown engine.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { DepthwiseChannelLab, DepthwiseRankLab, DepthwiseStencilLab, DepthwiseCoverageLab, DepthwiseContextLab, DepthwiseDigitLab, DepthwiseProgram, DepthwiseSupportingFigure } from '../../components/lesson-labs/DepthwiseConvolutionLabs.jsx';
export default {
  title: 'Depthwise Separable & Dilated Convolutions',
  readTime: '~65 min read + experiments and practice',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson depthwise-lesson"><LessonIntro prerequisites="Weighted sums and the preceding convolution lesson; spatial and channel axes, rank and receptive-field units are refreshed locally." sections={[["1-separate-the-spatial-and-channel-questions","1. Separate the spatial and channel questions"],["2-what-is-saved-and-what-is-restricted","2. What is saved—and what is restricted?"],["3-dilation-changes-positions-not-the-number-of-learned-taps","3. Dilation changes positions, not the number of learned taps"],["4-implement-the-operator-you-actually-mean","4. Implement the operator you actually mean"],["5-put-the-two-mechanisms-into-useful-architectures","5. Put the two mechanisms into useful architectures"],["6-a-real-experiment-compress-a-trained-layer-then-inspect-the-damage","6. A real experiment: compress a trained layer, then inspect the damage"],["reuse-the-spatial-operator-and-own-the-factorization","Reuse the spatial operator and own the factorization"],["7-diagnose-before-adding-another-architectural-feature","7. Diagnose before adding another architectural feature"],["8-practice-on-changed-problems","8. Practice on changed problems"],["9-readiness-and-the-next-design-question","9. Readiness and the next design question"]]}>Separate spatial filtering, channel mixing and sampling reach; inspect their costs and an actual compressed model.</LessonIntro>
<Prose>{""}<strong>{"Explore as you read."}</strong>{" Edit depthwise/pointwise filter entries, channel cells, stencil dilation/offsets, serial rates and parallel branch choices; manipulate retained digit inputs where weights are available. Update output contributions, rank restrictions, visited lattice sites, branch union and exact frozen-model outputs. Keep coverage geometry separate from learned influence. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose separability, dilation or parallel context from expressiveness, blind spots and the measured budget."}</Prose>

<Prose>{"A convolution usually answers two questions at once: "}<strong>{"what pattern appears nearby, and how should evidence from different channels be combined?"}</strong>{" A depthwise separable layer separates those jobs. A dilated layer changes a different choice: "}<strong>{"how far apart should the sampled positions be?"}</strong>{""}</Prose>

<Prose>{"These choices matter when a camera model must fit a device budget, when a segmentation model needs both local boundaries and distant context, or when you want to understand why a seemingly cheaper replacement changes predictions. The previous "}<a href={"/learn/path/full-curriculum/landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet?module=deep-learning-fundamentals"}>{"Landmark Architectures lesson"}</a>{" compared complete CNN designs. Here we open two of their building blocks and follow the actual numbers."}</Prose>

<Prose>{""}<strong>{"First pass:"}</strong>{" read sections 1–4 to understand the operators, 6–7 for a practical experiment and diagnosis, then attempt practice 1–5. Section 5 connects the mechanisms to mobile models and segmentation. The rank proof, gradient derivation and interval-coverage argument deepen the explanation; you can return to them without delaying the basic investigations. You need weighted sums and the idea that training changes weights to reduce a loss. Shapes, channel mixing and sampling geometry are refreshed here."}</Prose>

<H2>{"1. Separate the spatial and channel questions"}</H2>

<Prose>{"An image-like tensor has height, width and channels. RGB channels are measured colors; later channels are learned feature maps. In the PyTorch examples the axes are "}<strong>{"batch, channels, height, width"}</strong>{", written "}<InlineMath>{"N\\times C\\times H\\times W"}</InlineMath>{". A channel is a whole map, not one pixel or one class."}</Prose>

<Prose>{"For a standard "}<InlineMath>{"3\\times3"}</InlineMath>{" convolution, each output channel has a separate "}<InlineMath>{"3\\times3"}</InlineMath>{" filter for "}<strong>{"every"}</strong>{" input channel. At one output location it multiplies each local patch by the corresponding filter, sums within patches, then adds across channels. With "}<InlineMath>{"C_{\\rm in}"}</InlineMath>{" input and "}<InlineMath>{"C_{\\rm out}"}</InlineMath>{" output channels there are "}<InlineMath>{"9C_{\\rm in}C_{\\rm out}"}</InlineMath>{" weights, before biases."}</Prose>

<Prose>{"Imagine two input maps: one describes vertical contrast and another brightness. Standard convolution can use a different spatial treatment of each map for each output. A depthwise separable layer instead:"}</Prose>

<ol><li>{"Applies a spatial filter to each input channel independently. This is the "}<strong>{"depthwise"}</strong>{" step."}</li><li>{"Combines the resulting channel values at each location with learned weights. This is a "}<strong>{"pointwise"}</strong>{", or "}<InlineMath>{"1\\times1"}</InlineMath>{", convolution."}</li></ol>

<Prose>{"The pointwise operation has no spatial reach by itself. It can still perform substantial computation because it mixes all channels at every location. “One by one” describes its spatial kernel, not its channel connectivity."}</Prose>

<Prose>Follow one channel through its own spatial filter, then combine the two channel results. The channel investigation below exposes each scalar term beside the editable input.</Prose>

<H3>{"A complete forward calculation"}</H3>

<Prose>{"Use one spatial dimension so every multiplication fits on the page. Take two three-value channels, two-tap filters, stride one and no padding:"}</Prose>

<NeuralTable caption={"A complete forward calculation"} headers={[<>{"Quantity"}</>,<>{"Channel A"}</>,<>{"Channel B"}</>]} rows={[[<>{"Input"}</>,<>{""}<InlineMath>{"[1,2,3]"}</InlineMath>{""}</>,<>{""}<InlineMath>{"[2,0,1]"}</InlineMath>{""}</>],[<>{"Depthwise filter"}</>,<>{""}<InlineMath>{"[1,-1]"}</InlineMath>{""}</>,<>{""}<InlineMath>{"[0.5,1]"}</InlineMath>{""}</>],[<>{"First filtered value"}</>,<>{""}<InlineMath>{"1(1)+2(-1)=-1"}</InlineMath>{""}</>,<>{""}<InlineMath>{"2(0.5)+0(1)=1"}</InlineMath>{""}</>],[<>{"Second filtered value"}</>,<>{""}<InlineMath>{"2(1)+3(-1)=-1"}</InlineMath>{""}</>,<>{""}<InlineMath>{"0(0.5)+1(1)=1"}</InlineMath>{""}</>]]} />

<Prose>{"Let the output's pointwise weights be "}<InlineMath>{"[2,-1]"}</InlineMath>{". Its two values are both "}<InlineMath>{"2(-1)-1(1)=-3"}</InlineMath>{". The spatial filters made local summaries; the pointwise weights decided how those summaries interact. A negative pointwise weight subtracts evidence rather than discarding a channel."}</Prose>

<Prose>{"Try changing B's first input from 2 to 4. The first filtered B value becomes 2 and the first output becomes −4. The second output stays −3 because that input lies outside its patch. This is a useful locality check: not every edit should change every output."}</Prose>

<DepthwiseChannelLab />

<Prose>{"For two-dimensional filters without an intermediate activation:"}</Prose>

<div className="neural-equation"><MathBlock>{"z_c(i,j)=\\sum_{u,v}D_c(u,v)x_c(i+u,j+v),\\qquad\ny_o(i,j)=b_o+\\sum_c P_{o,c}z_c(i,j)."}</MathBlock></div>

<Prose>{""}<InlineMath>{"D"}</InlineMath>{" holds spatial weights, "}<InlineMath>{"P"}</InlineMath>{" holds channel-mixing weights, and "}<InlineMath>{"b"}</InlineMath>{" is an output bias. Index offsets above suppress padding and dilation for readability; section 3 restores them. Frameworks normally implement "}<strong>{"cross-correlation"}</strong>{", using weights in their stored order. Mathematical convolution reverses the kernel; learned kernels make either convention usable, but hand calculations must match the chosen one."}</Prose>

<H3>{"How its weights learn"}</H3>

<Prose>{"Training differentiates through both steps. For the first output above, target 1 and loss "}<InlineMath>{"L=\\frac12(y-1)^2"}</InlineMath>{", we have "}<InlineMath>{"y=-3"}</InlineMath>{", "}<InlineMath>{"L=8"}</InlineMath>{" and "}<InlineMath>{"\\partial L/\\partial y=-4"}</InlineMath>{"."}</Prose>

<Prose>{"The mixing gradient is "}<InlineMath>{"(-4)[-1,1]=[4,-4]"}</InlineMath>{". The gradient reaching filtered channels is "}<InlineMath>{"(-4)[2,-1]=[-8,4]"}</InlineMath>{". Multiplying these by the input patches gives spatial gradients A "}<InlineMath>{"[-8,-16]"}</InlineMath>{" and B "}<InlineMath>{"[8,0]"}</InlineMath>{"."}</Prose>

<Prose>{"A simultaneous gradient step of size 0.01 changes the mixing weights to "}<InlineMath>{"[1.96,-0.96]"}</InlineMath>{" and the spatial filters to "}<InlineMath>{"[1.08,-0.84]"}</InlineMath>{", "}<InlineMath>{"[0.42,1]"}</InlineMath>{". The new filtered values are −0.6 and 0.84; the output is −1.9824 and the loss is 4.44735488. Both sets of weights contribute to the improvement. This exact example is checked in the supplied author calculations. Larger steps need not reduce loss: a gradient describes a local direction, not a promise for every step size."}</Prose>

<Prose>{"With many locations, the gradient for a shared filter sums contributions from all its uses. With nonlinear activations, their derivatives also enter this chain."}</Prose>

<H2>{"2. What is saved—and what is restricted?"}</H2>

<Prose>{"At one output location, standard convolution uses "}<InlineMath>{"k^2C_{\\rm in}C_{\\rm out}"}</InlineMath>{" multiply-accumulates, or "}<strong>{"MACs"}</strong>{". One MAC multiplies two numbers and accumulates their product. Our convention counts it as one MAC; a convention counting separate floating-point operations often counts it as two FLOPs."}</Prose>

<Prose>{"With one depthwise filter per input channel, the depthwise and pointwise counts are:"}</Prose>

<div className="neural-equation"><MathBlock>{"k^2C_{\\rm in}+C_{\\rm in}C_{\\rm out}."}</MathBlock></div>

<Prose>{"Multiply either expression by "}<InlineMath>{"H_{\\rm out}W_{\\rm out}"}</InlineMath>{" for one image. These counts exclude bias additions, activation, normalization and memory movement. They equal the number of weights before multiplying by spatial area."}</Prose>

<Prose>{"For a "}<InlineMath>{"14\\times14"}</InlineMath>{" output, 64 input channels, 128 output channels and "}<InlineMath>{"3\\times3"}</InlineMath>{" filters:"}</Prose>

<NeuralTable caption={"2. What is saved—and what is restricted?"} headers={[<>{"Layer"}</>,<>{"Spatial weights"}</>,<>{"Mixing weights"}</>,<>{"Total weights"}</>,<>{"MACs/image"}</>]} rows={[[<>{"Standard"}</>,<>{"73,728"}</>,<>{"Included"}</>,<>{"73,728"}</>,<>{"14,450,688"}</>],[<>{"Depthwise + pointwise"}</>,<>{"576"}</>,<>{"8,192"}</>,<>{"8,768"}</>,<>{"1,718,528"}</>]]} />

<Prose>{"The separable-to-standard ratio is"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{1}{C_{\\rm out}}+\\frac{1}{k^2}."}</MathBlock></div>

<Prose>{"Here it is about 0.119. The pointwise operation now owns most arithmetic. The ratio approaches "}<InlineMath>{"1/9"}</InlineMath>{" for a "}<InlineMath>{"3\\times3"}</InlineMath>{" filter with very many output channels; it does not become exactly "}<InlineMath>{"1/9"}</InlineMath>{" at some magic channel threshold. For one output channel the ratio is "}<InlineMath>{"1+1/9"}</InlineMath>{": a separable factorization can cost "}<strong>{"more"}</strong>{"."}</Prose>

<Prose>{"These are exact operation counts under stated shapes, not measured speed. A runtime must read tensors, schedule kernels and exploit a particular processor's parallelism. A fast dense kernel can outperform an inefficiently implemented depthwise pipeline. Measure the actual deployment shape, batch, dtype and device before making a latency claim."}</Prose>

<DepthwiseSupportingFigure kind="cost" />

<H3>{"Why one filter per channel cannot represent every standard layer"}</H3>

<Prose>{"Substitute the first equation into the second. The effective standard kernel is"}</Prose>

<div className="neural-equation"><MathBlock>{"W_{o,c,u,v}=P_{o,c}D_{c,u,v}."}</MathBlock></div>

<Prose>{"Fix one input channel "}<InlineMath>{"c"}</InlineMath>{". Every output's spatial filter is a multiple of the "}<strong>{"same"}</strong>{" pattern "}<InlineMath>{"D_c"}</InlineMath>{". Standard convolution imposes no such restriction."}</Prose>

<Prose>{"A tiny counterexample makes the consequence visible. For one input channel with a two-value patch "}<InlineMath>{"[a,b]"}</InlineMath>{", suppose output 1 must equal "}<InlineMath>{"a"}</InlineMath>{" and output 2 must equal "}<InlineMath>{"b"}</InlineMath>{". The desired filter matrix is"}</Prose>

<div className="neural-equation"><MathBlock>{"W_c=\\begin{bmatrix}1&0\\\\0&1\\end{bmatrix}."}</MathBlock></div>

<Prose>{"One shared spatial filter cannot be both "}<InlineMath>{"[1,0]"}</InlineMath>{" and "}<InlineMath>{"[0,1]"}</InlineMath>{" after scalar rescaling. One possible rank-one approximation keeps only the first row: on patch "}<InlineMath>{"[2,3]"}</InlineMath>{" it produces "}<InlineMath>{"[2,0]"}</InlineMath>{" instead of "}<InlineMath>{"[2,3]"}</InlineMath>{". On "}<InlineMath>{"[2,0]"}</InlineMath>{" the same approximation happens to agree exactly. Agreement on one input does not prove equivalence of layers."}</Prose>

<Prose>{""}<strong>{"Rank"}</strong>{" counts independent patterns needed to express a matrix. This matrix has rank two. A "}<strong>{"depth multiplier"}</strong>{" "}<InlineMath>{"m"}</InlineMath>{" gives each input channel "}<InlineMath>{"m"}</InlineMath>{" different spatial filters, followed by mixing all "}<InlineMath>{"mC_{\\rm in}"}</InlineMath>{" intermediate channels:"}</Prose>

<div className="neural-equation"><MathBlock>{"W_{o,c,u,v}=\\sum_{r=1}^{m}P_{o,c,r}D_{c,r,u,v}."}</MathBlock></div>

<Prose>{"For each input channel, this permits rank at most "}<InlineMath>{"m"}</InlineMath>{". Any standard kernel can be represented when "}<InlineMath>{"m"}</InlineMath>{" reaches the largest of these per-channel ranks, which is no more than "}<InlineMath>{"\\min(C_{\\rm out},k^2)"}</InlineMath>{". Its weight count becomes "}<InlineMath>{"mC_{\\rm in}(k^2+C_{\\rm out})"}</InlineMath>{"; increasing expressiveness spends some or all of the savings."}</Prose>

<DepthwiseRankLab />

<H3>{"Two important distinctions"}</H3>

<Prose>{"A "}<InlineMath>{"3\\times1"}</InlineMath>{" filter followed by a "}<InlineMath>{"1\\times3"}</InlineMath>{" filter factors the "}<strong>{"spatial axes"}</strong>{". Depthwise followed by pointwise factors "}<strong>{"spatial treatment and channel mixing"}</strong>{". They are different restrictions and may be combined."}</Prose>

<Prose>{"An activation between the two operations changes the function. Even the scalar mapping "}<InlineMath>{"x\\mapsto\\max(0,x)"}</InlineMath>{" cannot be replaced by a fixed linear weight for both positive and negative inputs. The effective-kernel and rank equations above describe the linear pair, not a whole nonlinear MobileNet block. "}<a href={"https://arxiv.org/abs/1704.04861"}>{"MobileNet V1"}</a>{" places normalization and ReLU after both operations; "}<a href={"https://arxiv.org/abs/1610.02357"}>{"Xception's activation experiment"}</a>{" studies a different placement and finds an intermediate activation harmful in that setting. Architecture-specific evidence does not establish a universal activation rule."}</Prose>

<H2>{"3. Dilation changes positions, not the number of learned taps"}</H2>

<Prose>{"A three-tap filter normally samples positions "}<InlineMath>{"i-1,i,i+1"}</InlineMath>{". At "}<strong>{"dilation two"}</strong>{", it samples "}<InlineMath>{"i-2,i,i+2"}</InlineMath>{". The filter still has three weights. Dilation increases their spacing."}</Prose>

<Prose>{"With signal "}<InlineMath>{"[1,2,3,4,5,6,7,8,9]"}</InlineMath>{", center index 4 and filter "}<InlineMath>{"[1,1,1]"}</InlineMath>{":"}</Prose>

<NeuralTable caption={"3. Dilation changes positions, not the number of learned taps"} headers={[<>{"Dilation"}</>,<>{"Sampled indices"}</>,<>{"Sampled values"}</>,<>{"Sum"}</>]} rows={[[<>{"1"}</>,<>{"3, 4, 5"}</>,<>{"4, 5, 6"}</>,<>{"15"}</>],[<>{"2"}</>,<>{"2, 4, 6"}</>,<>{"3, 5, 7"}</>,<>{"15"}</>],[<>{"8, with zero padding"}</>,<>{"−4, 4, 12"}</>,<>{"0, 5, 0"}</>,<>{"5"}</>]]} />

<Prose>{"The first two sums agree because this particular signal is an arithmetic progression and the symmetric weights balance around the center. Their sampling patterns are still different. Change index 6 from 7 to 20: dilation two's sum becomes 28, while dilation one's sum stays 15. Change index 5 from 6 to 20 instead: dilation two stays 15, while dilation one becomes 29."}</Prose>

<DepthwiseStencilLab />

<Prose>{"For one spatial axis, input length "}<InlineMath>{"H"}</InlineMath>{", kernel size "}<InlineMath>{"k"}</InlineMath>{", dilation "}<InlineMath>{"d"}</InlineMath>{", symmetric padding "}<InlineMath>{"p"}</InlineMath>{" and stride "}<InlineMath>{"s"}</InlineMath>{":"}</Prose>

<div className="neural-equation"><MathBlock>{"H_{\\rm out}=\\left\\lfloor\\frac{H+2p-d(k-1)-1}{s}\\right\\rfloor+1."}</MathBlock></div>

<Prose>{"The kernel's "}<strong>{"bounding span"}</strong>{" is "}<InlineMath>{"k_{\\rm eff}=1+d(k-1)"}</InlineMath>{". A "}<InlineMath>{"3\\times3"}</InlineMath>{" kernel at dilation two spans a "}<InlineMath>{"5\\times5"}</InlineMath>{" box but directly reads only nine sites. At stride one, an odd kernel preserves size with "}<InlineMath>{"p=d(k-1)/2"}</InlineMath>{". For a "}<InlineMath>{"3\\times3"}</InlineMath>{" kernel, that is simply "}<InlineMath>{"p=d"}</InlineMath>{"."}</Prose>

<Prose>{"Stride moves the "}<strong>{"output centers"}</strong>{"; dilation spaces the "}<strong>{"taps within each output's stencil"}</strong>{". Zero padding supplies values outside the input. A dilated layer can also be depthwise: one choice concerns channel connectivity, the other concerns spatial sampling."}</Prose>

<Prose>{"At fixed output dimensions, changing dilation does not change the number of kernel weights or MAC terms. With no padding, larger dilation shrinks the output, so even the operation count changes. Always state the shape policy before comparing cost."}</Prose>

<H3>{"Receptive field, jump and actual coverage"}</H3>

<Prose>{"Suppose incoming features have receptive-field bounding width "}<InlineMath>{"r"}</InlineMath>{" and neighboring centers are "}<InlineMath>{"j"}</InlineMath>{" original-input positions apart. Adding a layer gives"}</Prose>

<div className="neural-equation"><MathBlock>{"r_{\\rm new}=r+(k-1)dj,\\qquad j_{\\rm new}=sj."}</MathBlock></div>

<Prose>{"Starting at "}<InlineMath>{"r=j=1"}</InlineMath>{", two "}<InlineMath>{"3\\times3"}</InlineMath>{" stride-one layers with dilations 1 and 2 produce width "}<InlineMath>{"1+2+4=7"}</InlineMath>{". A stride-two layer increases the jump for later layers; that is why simply adding all kernel spans fails after downsampling."}</Prose>

<DepthwiseSupportingFigure kind="field" />

<Prose>{"The bounding width says how far apart the extreme reachable positions are. It does not prove that every enclosed position is reachable, that learned weights are nonzero, or that a particular input has a nonzero gradient through every path. Those are different questions."}</Prose>

<Prose>{"For serial stride-one three-tap layers on an unbounded line, compute exact reachable offsets by starting with "}<InlineMath>{"\\{0\\}"}</InlineMath>{" and repeatedly adding each layer's set "}<InlineMath>{"\\{-d,0,d\\}"}</InlineMath>{". In two dimensions with full "}<InlineMath>{"3\\times3"}</InlineMath>{" stencils, take the Cartesian product of the one-dimensional set with itself."}</Prose>

<NeuralTable caption={"Receptive field, jump and actual coverage"} headers={[<>{"Serial dilation rates"}</>,<>{"Bounding width"}</>,<>{"Reachable 1D sites"}</>,<>{"Reachable 2D sites / box"}</>]} rows={[[<>{"2, 2, 2"}</>,<>{"13"}</>,<>{"7"}</>,<>{"49 / 169"}</>],[<>{"1, 2, 4"}</>,<>{"15"}</>,<>{"15"}</>,<>{"225 / 225"}</>],[<>{"1, 4"}</>,<>{"11"}</>,<>{"9"}</>,<>{"81 / 121"}</>],[<>{"1, 2, 5"}</>,<>{"17"}</>,<>{"17"}</>,<>{"289 / 289"}</>],[<>{"1, 2, 9"}</>,<>{"25"}</>,<>{"21"}</>,<>{"441 / 625"}</>]]} />

<Prose>{"Repeated even rates never connect an output to odd offsets. This is "}<strong>{"gridding"}</strong>{". But “the rates have greatest common divisor one” is not enough to remove all holes: rates 1 and 4 miss offsets −2 and 2. Their gcd is one. Check the reachable set instead of replacing geometry with a slogan. The original "}<a href={"https://arxiv.org/abs/1702.08502"}>{"hybrid dilated convolution paper"}</a>{" also imposes a gap condition; a warning against shared factors is not its complete construction rule."}</Prose>

<DepthwiseCoverageLab />

<details>

<summary>Deeper: construct a gap-free schedule and see its limit</summary>

<Prose>{"Suppose all offsets from "}<InlineMath>{"-S"}</InlineMath>{" through "}<InlineMath>{"S"}</InlineMath>{" are reachable. A new three-tap layer of rate "}<InlineMath>{"d"}</InlineMath>{" makes three shifted intervals centered at "}<InlineMath>{"-d,0,d"}</InlineMath>{". They join without missing integers if "}<InlineMath>{"d\\leq2S+1"}</InlineMath>{". The new covered interval is "}<InlineMath>{"[-S-d,S+d]"}</InlineMath>{"."}</Prose>

<Prose>{"Starting with "}<InlineMath>{"S=0"}</InlineMath>{" forces the first rate to be 1 for this construction. Choosing the largest permitted rate each time gives rates "}<InlineMath>{"1,3,9,27,\\ldots"}</InlineMath>{" and bounding widths "}<InlineMath>{"3,9,27,81,\\ldots"}</InlineMath>{". This is the same place-value idea as representing offsets with digits −1, 0 and 1 in powers of three."}</Prose>

<Prose>{"This is an exact sufficient construction for the stated unbounded, stride-one setting. It is not a recommendation to use enormous rates on tiny images. On a finite map, boundary padding can consume almost all the outer taps. Nor does complete structural coverage imply equal influence or better accuracy."}</Prose>

</details>

<H2>{"4. Implement the operator you actually mean"}</H2>

<Prose>{"In PyTorch, grouped convolution partitions input and output channels into separate groups. Both channel counts must be divisible by the group count. Depthwise convolution uses one group per input channel. With multiplier "}<InlineMath>{"m"}</InlineMath>{", it produces "}<InlineMath>{"mC_{\\rm in}"}</InlineMath>{" channels; a following ordinary "}<InlineMath>{"1\\times1"}</InlineMath>{" layer mixes them."}</Prose>

<Prose>{"This complete program runs a linear depthwise/pointwise pair, constructs its effective standard kernel, and checks equality:"}</Prose>

<CodeBlock language={"python"}>{"import torch\nfrom torch import nn\nfrom torch.nn import functional as F\n\ntorch.manual_seed(4)\ntorch.set_num_threads(1)\ninputs, outputs, multiplier, dilation = 3, 5, 2, 2\nimage = torch.randn(1, inputs, 7, 7, dtype=torch.float64)\ndepthwise = nn.Conv2d(\n    inputs, inputs * multiplier, 3, padding=dilation,\n    dilation=dilation, groups=inputs, bias=False\n).double()\npointwise = nn.Conv2d(inputs * multiplier, outputs, 1).double()\n\nfiltered = depthwise(image)\nseparated = pointwise(filtered)\nspatial = depthwise.weight[:, 0].reshape(inputs, multiplier, 3, 3)\nmixing = pointwise.weight[:, :, 0, 0].reshape(outputs, inputs, multiplier)\neffective = torch.einsum(\"ocm,cmuv->ocuv\", mixing, spatial)\nstandard = F.conv2d(image, effective, pointwise.bias,\n                    padding=dilation, dilation=dilation)\nprint(filtered.shape, separated.shape)\nprint(torch.allclose(separated, standard, atol=1e-12, rtol=1e-12))"}</CodeBlock>

<Prose>{"The intermediate shape is "}<InlineMath>{"1\\times6\\times7\\times7"}</InlineMath>{", the output is "}<InlineMath>{"1\\times5\\times7\\times7"}</InlineMath>{", and the equality check is true within floating-point tolerance. The construction sums the "}<InlineMath>{"m"}</InlineMath>{" shared spatial patterns for each input/output channel pair; it does not train or approximate anything."}</Prose>

<Prose>{"The complete "}<a href={"/learn-code/depthwise-separable-dilated-convolutions/author-checks.py"}>{"direct-loop reference and checks"}</a>{" also implement grouping, padding, dilation and stride without calling a convolution library inside the reference. Nine float64 comparisons against PyTorch, across groups 1/2/4 and several dilation/stride settings, had maximum absolute error "}<InlineMath>{"5.33\\times10^{-15}"}</InlineMath>{". That validates these fixtures, rather than claiming every future dtype/device/input is tested. The stable "}<a href={"https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html"}>{"PyTorch Conv2d API reference"}</a>{" defines the same group and output-shape conventions."}</Prose>

<DepthwiseProgram file="author-checks.py" title="Read the direct-loop spatial operator and verification program" />

<H3>{"Biases, normalization and small batches"}</H3>

<Prose>{"If the depthwise operation has bias "}<InlineMath>{"b_D"}</InlineMath>{" and the pointwise operation bias "}<InlineMath>{"b_P"}</InlineMath>{", combining the linear pair produces bias "}<InlineMath>{"Pb_D+b_P"}</InlineMath>{". Ignoring the first bias changes the function."}</Prose>

<Prose>{"At evaluation, a BatchNorm channel uses fixed stored mean "}<InlineMath>{"\\mu"}</InlineMath>{" and variance "}<InlineMath>{"v"}</InlineMath>{". A preceding convolution with weights "}<InlineMath>{"W"}</InlineMath>{" and bias "}<InlineMath>{"b"}</InlineMath>{" can absorb this affine transformation:"}</Prose>

<div className="neural-equation"><MathBlock>{"W'=\\frac{\\gamma W}{\\sqrt{v+\\epsilon}},\\qquad\nb'=\\beta+\\frac{\\gamma(b-\\mu)}{\\sqrt{v+\\epsilon}}."}</MathBlock></div>

<Prose>{"This is inference folding. Training BatchNorm depends on current batch statistics, so the same fixed-folding argument does not apply. A global-pooling branch with one image and a "}<InlineMath>{"1\\times1"}</InlineMath>{" map has only one value per channel; training-mode BatchNorm cannot estimate its usual channel variance from that singleton. Use an appropriate trained inference state or deliberately choose a different normalization design; changing only gradient tracking does not change module mode."}</Prose>

<H2>{"5. Put the two mechanisms into useful architectures"}</H2>

<H3>{"Mobile networks: spend channel mixing carefully"}</H3>

<Prose>{"MobileNet V1 repeatedly applies depthwise spatial filtering followed by pointwise mixing. Reducing every internal channel width by a factor "}<InlineMath>{"\\alpha"}</InlineMath>{" reduces the depthwise term linearly and the pointwise term quadratically:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\text{MACs}\\approx H_{\\rm out}W_{\\rm out}\n\\left(k^2\\alpha C_{\\rm in}+\\alpha^2C_{\\rm in}C_{\\rm out}\\right)."}</MathBlock></div>

<Prose>{"Reducing both spatial dimensions by "}<InlineMath>{"\\rho"}</InlineMath>{" multiplies this by approximately "}<InlineMath>{"\\rho^2"}</InlineMath>{", subject to integer rounding and boundary stages. Width and resolution are different choices: narrowing removes feature capacity, whereas downsampling can remove fine spatial evidence. The broad architecture comparison in the previous lesson now has a mechanistic explanation."}</Prose>

<Prose>{""}<a href={"https://arxiv.org/abs/1801.04381"}>{"MobileNet V2"}</a>{" expands a narrow representation with a pointwise operation, performs depthwise spatial work in the expanded space, and projects back without a final clipping activation in the branch. When shapes match, a residual path connects the narrow endpoints. Expansion provides multiple nonlinear features before compression; the linear projection avoids obligatorily zeroing every negative projected value."}</Prose>

<Prose>{"For equal input/output width "}<InlineMath>{"C"}</InlineMath>{", expansion factor "}<InlineMath>{"t"}</InlineMath>{", stride one and a "}<InlineMath>{"k\\times k"}</InlineMath>{" spatial operation, branch weights before biases/normalization are"}</Prose>

<div className="neural-equation"><MathBlock>{"tC^2+k^2tC+tC^2."}</MathBlock></div>

<Prose>{"Expansion is not free. At "}<InlineMath>{"C=32,t=6,k=3"}</InlineMath>{" there are 14,016 weights and 2,747,136 MACs on a "}<InlineMath>{"14\\times14"}</InlineMath>{" map. If the depthwise operation downsamples to "}<InlineMath>{"7\\times7"}</InlineMath>{", expansion still runs on "}<InlineMath>{"14\\times14"}</InlineMath>{", while spatial filtering and projection run on "}<InlineMath>{"7\\times7"}</InlineMath>{"; the count becomes 1,589,952. Multiplying every operation by the smaller area would undercount the block."}</Prose>

<Prose>{"MobileNet V3 adds design choices including channel gates, architecture search and hard-swish. Its gate approximation "}<InlineMath>{"\\operatorname{clip}(x+3,0,6)/6"}</InlineMath>{" is piecewise linear; multiplying by "}<InlineMath>{"x"}</InlineMath>{" gives "}<strong>{"hard-swish"}</strong>{", which is piecewise quadratic in the middle:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\operatorname{hswish}(x)=\n\\begin{cases}\n0 & x\\leq-3,\\\\\nx(x+3)/6 & -3<x<3,\\\\\nx & x\\geq3.\n\\end{cases}"}</MathBlock></div>

<Prose>{"For example, hard-swish(−2)=−2/3. It preserves some negative responses, unlike ReLU. Deployment benefit depends on an implementation and processor, not merely on avoiding a sigmoid. The exact design and measured platform belong to the "}<a href={"https://arxiv.org/abs/1905.02244"}>{"MobileNet V3 paper"}</a>{", not a universal speed rule."}</Prose>

<DepthwiseSupportingFigure kind="mobile" />

<H3>{"Segmentation: ask for context at every location"}</H3>

<Prose>{"Classification returns one label per image. "}<strong>{"Semantic segmentation"}</strong>{" assigns a class to each pixel. A patch may resemble road or roof locally; its surroundings help distinguish them. But repeated downsampling can erase a thin pole before the final prediction."}</Prose>

<Prose>{"Dilation lets later layers sample more distant context while keeping their output grid dense. "}<strong>{"Output stride"}</strong>{" is the spacing between output-feature centers in original-image coordinates. If it is 16, a feature-map dilation of 6 spaces adjacent taps "}<InlineMath>{"6\\times16=96"}</InlineMath>{" input pixels apart. If output stride changes to 8, dilation 12 keeps that spacing. The complete receptive-field size still depends on the backbone's incoming field; tap spacing alone does not specify it."}</Prose>

<Prose>{"DeepLab V3's "}<strong>{"atrous spatial pyramid pooling"}</strong>{", or ASPP, processes one map through parallel branches: a pointwise branch, several dilated "}<InlineMath>{"3\\times3"}</InlineMath>{" branches, and an image-summary branch. Concatenation preserves their separate channels; a learned pointwise projection combines them. The original V3 setting uses rates 6/12/18 at output stride 16, with doubled rates at output stride 8. These are "}<strong>{"parallel"}</strong>{" branches, so applying the serial gridding formula to the list 6/12/18 is a category error. "}<a href={"https://arxiv.org/abs/1706.05587"}>{"DeepLab V3"}</a>{""}</Prose>

<DepthwiseContextLab />

<Prose>{"The image-summary branch averages each channel over space, projects the vector and broadcasts/upsamples it back. It brings global context, but discards where that context occurred. Two spatial arrangements with the same channel means give the same image-summary vector."}</Prose>

<Prose>{"Large dilation can also fail on small feature maps. On an "}<InlineMath>{"8\\times8"}</InlineMath>{" map, a "}<InlineMath>{"3\\times3"}</InlineMath>{" kernel at dilation 8 has only its center tap inside the map at every output position. Most nominal taps multiply padding. This is still a learned operation, but it has lost the intended distant-image context. DeepLab V3 discusses this boundary effect; its global branch is one response."}</Prose>

<Prose>{"DeepLab V3+ adds a decoder that combines coarse semantic features with higher-resolution features and refines boundaries. Its use of atrous separable convolutions combines our two independent choices. The decoder does not magically reconstruct all information lost during downsampling; its skip features provide additional spatial evidence. "}<a href={"https://arxiv.org/abs/1802.02611"}>{"DeepLab V3+"}</a>{""}</Prose>

<Prose>{"These ideas also transfer to other domains. In an audio feature map, an axis may represent time and another frequency; dilation rates should reflect the corresponding units. For real-time sequence prediction a centered temporal stencil reads future samples. A causal version uses only current and earlier positions, with left padding; the recurrent and sequence lessons will develop that decision further. A large mathematical field is useful only when the sampled information is available and relevant."}</Prose>

<Prose>{"For a complete building-block program, run "}<a href={"/learn-code/depthwise-separable-dilated-convolutions/context-blocks.py"}>{"context-blocks.py"}</a>{" beside the supplied CSV. It defines an inverted residual block and an ASPP-style parallel context module, processes two real digit images through randomly initialized layers, and differentiates a scalar probe back to the stem. The mobile output is "}<InlineMath>{"2\\times8\\times8\\times8"}</InlineMath>{" and the context output "}<InlineMath>{"2\\times4\\times8\\times8"}</InlineMath>{". This demonstrates shape correspondence and gradient connectivity, not fitted segmentation accuracy. It uses rates 1 and 2 on the small map, four channels per context branch, and a projection input width derived from the actual number of branches. Two images keep the global branch's training-mode BatchNorm from receiving a singleton channel sample. The full segmentation system would additionally need an appropriate backbone, pixel labels, a supervised pixel loss and evaluation."}</Prose>

<DepthwiseProgram file="context-blocks.py" title="Read the inverted residual and parallel-context program" />

<H2>{"6. A real experiment: compress a trained layer, then inspect the damage"}</H2>

<Prose>{"Can a trained standard convolution be replaced by a cheaper linear depthwise/pointwise pair without retraining? This is a concrete model-compression question, different from training a new mobile architecture from scratch."}</Prose>

<Prose>{"We use 400 real optical digit images, 40 per label, from the UCI dataset distributed with scikit-learn. Each image is "}<InlineMath>{"8\\times8"}</InlineMath>{" with intensities 0–16; divide by 16. The "}<a href={"/learn-code/depthwise-separable-dilated-convolutions/digits-400.csv"}>{"attributed offline CSV"}</a>{" and "}<a href={"/learn-code/depthwise-separable-dilated-convolutions/data-provenance.md"}>{"provenance"}</a>{" make the experiment reproducible without a download. All 400 pixel vectors and source IDs were checked for exact duplicates before splitting. Writer IDs are unavailable, so this does not establish recognition of independent writers."}</Prose>

<Prose>{"The stratified split uses 280 training and 120 development images, seed 22. It is a teaching split inside a selected subset, not the original benchmark protocol and not an untouched final test. We inspect all candidates on development; a deployment claim would need a subsequent frozen evaluation."}</Prose>

<Prose>{"The model is deliberately small:"}</Prose>

<div className="neural-equation"><MathBlock>{"1\\times8\\times8\n\\xrightarrow{\\;3\\times3,\\ 8;\\ \\mathrm{ReLU}\\;}\n8\\times8\\times8\n\\xrightarrow{\\;3\\times3,\\ 12;\\ \\mathrm{ReLU}\\;}\n12\\times8\\times8\n\\xrightarrow{\\;\\mathrm{average\\ pool}\\ 2\\;}\n12\\times4\\times4\n\\xrightarrow{\\;\\mathrm{flatten,\\ linear}\\;}\n10\\ \\text{logits}."}</MathBlock></div>

<Prose>{"We train it separately with dilation 1 or 2 in the second convolution, always padding by the dilation to preserve shape. Within each of seeds 1/2/3, weights initially match across the two models. Both have 2,886 parameters and 61,824 convolution/linear MACs per image. Adam, learning rate 0.003, runs 400 full-batch updates with no augmentation, dropout, normalization, weight decay or early stopping. All six final models classify all 280 training images correctly."}</Prose>

<H3>{"Factor only the trained second layer"}</H3>

<Prose>{"For each input channel, flatten its 12 output filters into a "}<InlineMath>{"12\\times9"}</InlineMath>{" matrix. "}<strong>{"Singular value decomposition"}</strong>{", or SVD, writes this matrix as a sum of independent rank-one patterns, ordered by strength. Keeping the first "}<InlineMath>{"m"}</InlineMath>{" patterns gives the smallest squared error between original and reconstructed weights among rank-at-most-"}<InlineMath>{"m"}</InlineMath>{" matrices. This statement concerns the weights; it does not minimize classification loss or guarantee better predictions as "}<InlineMath>{"m"}</InlineMath>{" increases."}</Prose>

<Prose>{"The saved program puts the retained right singular vectors into depthwise filters and the scaled left vectors into pointwise mixing weights. It copies the old output bias to the pointwise layer, adds no activation between the factors, and retains the original ReLU after them. Everything else stays fixed. Multipliers 1, 2, 4 and 9 are declared before observing results. No compressed model is retrained."}</Prose>

<Prose>{"Download "}<a href={"/learn-code/depthwise-separable-dilated-convolutions/convolution-factorization.py"}>{"the complete training and factorization program"}</a>{" beside the CSV. In a Python environment with NumPy, scikit-learn and PyTorch:"}</Prose>

<CodeBlock language={"text"}>{"python convolution-factorization.py"}</CodeBlock>

<Prose>{"The program contains all imports, model definitions, split, training loop, SVD conversion, evaluation and result writing. It produces "}<a href={"/learn-code/depthwise-separable-dilated-convolutions/calculated-inputs.json"}>{"calculated-inputs.json"}</a>{", including all six runs, development predictions, exact model costs and two saved seed-one dense states. The following is the key factor construction explained in isolation; it is an excerpt from that complete program:"}</Prose>

<DepthwiseProgram file="convolution-factorization.py" title="Read the complete training and SVD factorization program" />

<CodeBlock language={"python"}>{"matrix = layer.weight[:, channel].reshape(outputs, -1)\nleft, values, right = torch.linalg.svd(matrix, full_matrices=False)\ndepthwise.weight[start:start + multiplier, 0] = (\n    right[:multiplier].reshape(multiplier, height, width)\n)\npointwise.weight[:, start:start + multiplier, 0, 0] = (\n    left[:, :multiplier] * values[:multiplier]\n)"}</CodeBlock>

<Prose>{"There is one such assignment per input channel. The intermediate channels are ordered as all retained filters for input channel 0, then channel 1, and so forth. Grouped convolution and the reshape must agree on that ordering."}</Prose>

<H3>{"Results from the actual CPU run"}</H3>

<Prose>{"Each entry below is the number correct out of the "}<strong>{"same 120 development images"}</strong>{":"}</Prose>

<NeuralTable caption={"Results from the actual CPU run"} headers={[<>{"Seed"}</>,<>{"Dilation"}</>,<>{"Original dense"}</>,<>{""}<InlineMath>{"m=1"}</InlineMath>{""}</>,<>{""}<InlineMath>{"m=2"}</InlineMath>{""}</>,<>{""}<InlineMath>{"m=4"}</InlineMath>{""}</>,<>{""}<InlineMath>{"m=9"}</InlineMath>{""}</>]} rows={[[<>{"1"}</>,<>{"1"}</>,<>{"117"}</>,<>{"112"}</>,<>{"115"}</>,<>{"117"}</>,<>{"117"}</>],[<>{"1"}</>,<>{"2"}</>,<>{"115"}</>,<>{"95"}</>,<>{"110"}</>,<>{"113"}</>,<>{"115"}</>],[<>{"2"}</>,<>{"1"}</>,<>{"116"}</>,<>{"90"}</>,<>{"112"}</>,<>{"116"}</>,<>{"116"}</>],[<>{"2"}</>,<>{"2"}</>,<>{"116"}</>,<>{"56"}</>,<>{"96"}</>,<>{"112"}</>,<>{"116"}</>],[<>{"3"}</>,<>{"1"}</>,<>{"115"}</>,<>{"107"}</>,<>{"114"}</>,<>{"115"}</>,<>{"115"}</>],[<>{"3"}</>,<>{"2"}</>,<>{"115"}</>,<>{"98"}</>,<>{"106"}</>,<>{"114"}</>,<>{"115"}</>]]} />

<NeuralTable caption={"Results from the actual CPU run"} headers={[<>{"Whole-model form"}</>,<>{"Parameters"}</>,<>{"Convolution/linear MACs/image"}</>]} rows={[[<>{"Original dense"}</>,<>{"2,886"}</>,<>{"61,824"}</>],[<>{""}<InlineMath>{"m=1"}</InlineMath>{""}</>,<>{"2,190"}</>,<>{"17,280"}</>],[<>{""}<InlineMath>{"m=2"}</InlineMath>{""}</>,<>{"2,358"}</>,<>{"28,032"}</>],[<>{""}<InlineMath>{"m=4"}</InlineMath>{""}</>,<>{"2,694"}</>,<>{"49,536"}</>],[<>{""}<InlineMath>{"m=9"}</InlineMath>{""}</>,<>{"3,534"}</>,<>{"103,296"}</>]]} />

<Prose>{"Rank one can make a drastic difference: seed 2 with dilation two drops from 116 to 56 correct. That is evidence against treating arbitrary depthwise replacement as function-preserving compression. It is not evidence that trained depthwise networks are inherently poor classifiers; we did not train these compressed networks for their new constraint."}</Prose>

<Prose>{"Rank nine recovers every per-channel "}<InlineMath>{"12\\times9"}</InlineMath>{" matrix, up to numerical error, and therefore recovers predictions. Its maximum absolute logit difference across all six runs is about "}<InlineMath>{"1.72\\times10^{-5}"}</InlineMath>{" in this float32 execution. It also costs more than the original layer. Exact reconstruction and worthwhile compression are distinct goals."}</Prose>

<Prose>{"Even equal accuracy can hide different confidence. Seed 1, dilation one has development cross-entropy 0.128695 before compression and 0.137114 at rank four, although both have 117 correct. Cross-entropy penalizes assigning low probability to the actual label; the count correct only records the largest logit. Inspect both."}</Prose>

<Prose>Inspect the saved seed-one dense model alongside its rank-one replacement on the same real image. Source 299 (digit 1) and source 32 (digit 9) are deliberately selected disagreement cases for dilations 1 and 2 respectively; they are diagnostic selections, not random representatives. Change a pixel to compare the actual signed logits and inspect original/reconstructed filters.</Prose><DepthwiseDigitLab />

<Prose>{"A sensible next experiment would fine-tune the compressed model using training data only, predeclare its update budget, then use development to assess the trade-off. Another would compare training the separable architecture from scratch. Neither experiment was run here, so the page does not supply imagined recovery curves. The six fits show a bounded mechanism and its variability, not a hardware benchmark or a general ranking of dilation rates."}</Prose>

<H2>{"Reuse the spatial operator and own the factorization"}</H2>

<Prose>{"There are two different implementation tasks here. The preceding "}<a href={"/learn-code/depthwise-separable-dilated-convolutions/../convolution-pooling-receptive-fields/convolution-experiments.py"}>{"convolution program, "}<code>{"direct_conv2d"}</code>{""}</a>{" owns address calculation for grouped/dilated filtering; its published "}<a href={"/learn-code/depthwise-separable-dilated-convolutions/../convolution-pooling-receptive-fields/convolution_pullbacks.py"}>{"pullback program"}</a>{" opens the batched dense derivatives. The improved "}<a href={"/learn/path/full-curriculum/convolution-pooling-receptive-fields?module=deep-learning-fundamentals"}>{"Convolution, Pooling & Receptive Fields lesson"}</a>{" is published. Reuse those actual operators while this lesson opens their factorization and sampling choices."}</Prose>

<Prose>{"This topic owns the new decomposition. In "}<a href={"/learn-code/depthwise-separable-dilated-convolutions/convolution-factorization.py"}>{"convolution-factorization.py"}</a>{", "}<code>{"factor_spatial"}</code>{" takes an existing kernel, performs one truncated SVD per input channel, writes the depthwise and pointwise weights, and preserves the output bias. "}<code>{"reconstructed_weight"}</code>{" independently contracts the factors back into the dense tensor. The "}<code>{"nn.Conv2d(groups=in_channels)"}</code>{" route is the normal tool implementation, with intermediate channel order fixed as "}<code>{"input_channel * multiplier + component"}</code>{". The script compares reconstructed weights, outputs and the unchanged full model before interpreting prediction changes. "}<code>{"context-blocks.py"}</code>{" separately composes the inverted residual and parallel-context mechanisms already explained above."}</Prose>

<Prose>{"For I inputs, O outputs and K spatial taps, factoring all full O×K matrices costs O(I·min(O,K)²·max(O,K)) for a dense SVD; storing factors costs O(Im(K+O)). The multiplication benefit depends on m; raising m to full rank can remove approximation error while costing more than the original layer. The SVD itself reuses the published "}<a href={"/learn/path/full-curriculum/matrix-decompositions-svd-qr-cholesky-lu"}>{"Matrix Decompositions owner"}</a>{", rather than making this lesson secretly responsible for a numerical factorization library."}</Prose>

<Prose>{""}<strong>{"Take control:"}</strong>{" add a relative discarded-energy report before replacing a layer. For each input matrix, divide the sum of squares of discarded singular values by the sum of squares of all singular values; define the all-zero matrix's relative error as zero. Select the smallest per-input m meeting an energy budget, then decide whether to pad all groups to a common m or implement heterogeneous groups separately. A fixed grouped Conv2d expects equal intermediate multiplicity, so silently assigning variable ranks to its regular tensor layout is wrong."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"The singular values already exist during factorization; use their squares, not their sum, for Frobenius reconstruction energy."}</Prose>

</details>

<details>

<summary>Solution and success criteria</summary>

<Prose>{"For singular values [4,3,0], rank1 leaves9/25=0.36 relative squared error; rank2 leaves zero. A0.1 budget therefore needs rank2. If two input channels need ranks1 and2, a regular multiplier2 representation keeps the second slot of the first channel zero, or a custom grouped implementation must explicitly carry different slices. Verify reconstructed-weight error against the spectral sum, preserve bias/dilation/padding, and measure task outcomes separately from weight energy."}</Prose>

</details>

<H2>{"7. Diagnose before adding another architectural feature"}</H2>

<Prose>{"If a separable replacement changes predictions, first check whether it was intended to be an exact algebraic rewrite or an approximation. Confirm intermediate activations, bias transfer and channel ordering. Then inspect rank error, calibration/error slices and the opportunity for training adaptation."}</Prose>

<Prose>{"If a dilation change seems ineffective, inspect actual sampled positions on the actual map. A linear ramp can hide the difference; oversized rates can sample mostly padding; repeated rates can leave unreachable offsets. The receptive-field outline alone cannot distinguish these cases."}</Prose>

<Prose>{"If inference is slow despite fewer MACs, profile the deployed graph. Separate measured elapsed time from operation estimates, and include normalization, activation, memory transfers and kernel overhead. Warm up the actual implementation; synchronize asynchronous devices before timing; report shape, batch, dtype, device and software. This lesson supplies no latency ranking."}</Prose>

<Prose>{"If grouping fails, inspect divisibility. A 96-input, 128-output layer with 32 groups is valid: each group has 3 inputs and 4 outputs. Changing 128 outputs to 130 is invalid. Rounding widths to multiples of 8 for a hardware convention does not automatically make them divisible by every chosen group count."}</Prose>

<Prose>{"If RGB handling worries you, remember that pointwise mixing can combine color-channel responses. The real restriction is the spatial/channel factorization, not an inability to use colors. Whether that restriction is appropriate in an input stem is an empirical design question, not a universal prohibition."}</Prose>

<H2>{"8. Practice on changed problems"}</H2>

<H3>{"1. Follow a changed channel through the layer"}</H3>

<Prose>{"Use input channels A "}<InlineMath>{"[2,1,0]"}</InlineMath>{", B "}<InlineMath>{"[1,3,2]"}</InlineMath>{", spatial filters A "}<InlineMath>{"[1,2]"}</InlineMath>{", B "}<InlineMath>{"[-1,1]"}</InlineMath>{", and pointwise weights "}<InlineMath>{"[0.5,2]"}</InlineMath>{". Find both outputs. Then change A's final value from 0 to 4: which output changes, and by how much?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Compute the two valid two-value patches in each channel before mixing. The final A value participates only in the second patch."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"A filters to "}<InlineMath>{"[4,1]"}</InlineMath>{" and B to "}<InlineMath>{"[2,-1]"}</InlineMath>{". Outputs are "}<InlineMath>{"[6,-1.5]"}</InlineMath>{". Editing A's last value changes its second filtered value from 1 to 9, so the second output increases by "}<InlineMath>{"0.5(8)=4"}</InlineMath>{" to 2.5. The first remains 6."}</Prose>

</details>

<H3>{"2. Budget expressiveness"}</H3>

<Prose>{"A "}<InlineMath>{"3\\times3"}</InlineMath>{" layer has 16 inputs and 24 outputs. Compare a dense layer with a separable pair using multiplier two. Ignore biases. Does a smaller parameter count prove equivalence or faster inference?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"The multiplier repeats spatial filters and increases the input width of the pointwise layer. Count both."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"Dense: "}<InlineMath>{"9(16)(24)=3456"}</InlineMath>{" weights. Separable: "}<InlineMath>{"16(2)(9+24)=1056"}</InlineMath>{". The effective per-input-channel rank is at most two, so an arbitrary dense layer need not be representable. Hardware execution and all other operations determine elapsed time."}</Prose>

</details>

<H3>{"3. Repair a misleading receptive field"}</H3>

<Prose>{"Two three-tap layers use rates 1 and 4. List reachable offsets, identify the gaps, and choose a replacement second rate that reaches every offset in a seven-position span."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Shift the first layer's set "}<InlineMath>{"\\{-1,0,1\\}"}</InlineMath>{" left, nowhere and right by the second rate."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"Rates 1 and 4 reach "}<InlineMath>{"\\{-5,-4,-3,-1,0,1,3,4,5\\}"}</InlineMath>{" and miss −2,2 inside their eleven-position bound. Second rate 2 produces every offset −3 through 3, a seven-position span. The repaired field is smaller but fully covered."}</Prose>

</details>

<H3>{"4. Respect original-image units"}</H3>

<Prose>{"Incoming features have receptive-field width 7 and jump 2 input pixels. Add a kernel of size 3, dilation 3 and stride 2. Find the new field width and jump. Explain why "}<InlineMath>{"2(2\\cdot3+1)"}</InlineMath>{" is not the field width."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"The new extreme taps enlarge an existing field; each incoming feature already summarizes several original pixels."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{""}<InlineMath>{"r'=7+(3-1)(3)(2)=19"}</InlineMath>{", and "}<InlineMath>{"j'=2(2)=4"}</InlineMath>{". Multiplying a seven-tap-span outline by the old jump gives 14, which ignores the incoming field width and conflates center spacing with the coverage of one feature."}</Prose>

</details>

<H3>{"5. Design the next compression experiment"}</H3>

<Prose>{"A rank-two replacement has worse development accuracy than the original, but fewer MACs. Propose an experiment that answers whether training adaptation can recover useful accuracy. Identify which data may drive updates and what evidence is still needed before a latency claim."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Fix what stays constant, predeclare what training changes, and separate the role of training examples from development examples."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"Start from the same saved dense model and the declared rank-two conversion. Predeclare an optimizer, update budget and which layers may change. Fine-tune only on training images; retain the original and unadapted compressed models as baselines. Compare development counts, cross-entropy and relevant slices after the fixed run, and report all chosen seeds. Selecting a strategy on development consumes that evidence; later reporting needs a frozen evaluation. MAC reduction still requires a measured deployment timing comparison with documented device, software and input contract."}</Prose>

</details>

<H3>{"6. A spatial summary can miss a rearrangement"}</H3>

<Prose>{"An ASPP image-summary branch averages a channel whose "}<InlineMath>{"2\\times2"}</InlineMath>{" values are "}<InlineMath>{"[1,3;5,7]"}</InlineMath>{". Swap the top-left and bottom-right values. Can this branch alone identify the swap? Could a local branch respond differently?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"The global branch sees a channel mean, while a local stencil sees values at particular offsets."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"The mean remains 4, so the pooled vector and its deterministic projection remain identical. A local branch may change because the values at its sampled positions changed. A particular symmetric filter or location could still give the same result; dependency permits a change but does not guarantee one."}</Prose>

</details>

<H3>{"7. An advanced construction challenge"}</H3>

<Prose>{"Using four serial three-tap, stride-one layers on an unbounded line, construct a gap-free schedule with the largest possible number of distinct reachable positions. State why the result is a structural upper bound, and why it may be unsuitable for an "}<InlineMath>{"8\\times8"}</InlineMath>{" feature map."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Each layer offers three tap choices per path. Use the interval construction to make those paths reach distinct offsets."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"There are "}<InlineMath>{"3^4=81"}</InlineMath>{" tap-choice paths, so at most 81 distinct positions can be reached. Rates 1,3,9,27 reach every integer from −40 to 40, attaining that bound. Finite maps invalidate the unbounded-input premise: many taps address padding, and the nominal span does not create new image information."}</Prose>

</details>

<H2>{"9. Readiness and the next design question"}</H2>

<Prose>{"You are ready to continue when you can trace spatial filtering and channel mixing separately; state the rank restriction of a linear separable pair; calculate weights and MACs with the correct output shapes; distinguish dilation from stride; and test sampled-site coverage instead of trusting a bounding rectangle. In practice, you should also be able to identify whether an architectural change was retrained, whether its evaluation data was already used for selection, and whether a speed claim was actually measured."}</Prose>

<Prose>{"Next in the module is "}<strong>{""}<a href={"/learn/path/full-curriculum/convnext-modern-cnn-designs?module=deep-learning-fundamentals"}>{"ConvNeXt & Modern CNN Designs"}</a>{""}</strong>{". It builds on this distinction between spatial mixing and channel mixing, then changes normalization, activation placement, stage design and training. We will ask what evidence supports those changes rather than assuming a newer name explains the improvement."}</Prose>

<H3>{"References and other ways to learn"}</H3>

<ul><li>{""}<a href={"https://arxiv.org/abs/1704.04861"}>{"MobileNets V1, Howard et al."}</a>{": sections 3.1–3.4 explain separable filtering and width/resolution choices. Read after sections 1–2; its historical model and reported costs have a specific shape/training context."}</li><li>{""}<a href={"https://arxiv.org/abs/1610.02357"}>{"Xception, Chollet"}</a>{": section 4.7 is a useful contrasting activation-placement experiment. Its conclusion is about the tested architecture."}</li><li>{""}<a href={"https://arxiv.org/abs/1801.04381"}>{"MobileNet V2, Sandler et al."}</a>{": sections 3.2–3.4 and the block table explain linear bottlenecks and inverted residuals. Its geometric motivation needs more linear algebra than the first-pass route."}</li><li>{""}<a href={"https://arxiv.org/abs/1905.02244"}>{"Searching for MobileNet V3, Howard et al."}</a>{": section 5 distinguishes the hard gate from hard-swish and documents architecture-specific deployment decisions."}</li><li>{""}<a href={"https://arxiv.org/abs/1511.07122"}>{"Multi-Scale Context Aggregation by Dilated Convolutions, Yu and Koltun"}</a>{": sections 2–3 introduce the operator and a context module; later sections separate the front end and evaluation. Useful after drawing exact tap positions."}</li><li>{""}<a href={"https://arxiv.org/abs/1702.08502"}>{"Understanding Convolution for Semantic Segmentation, Wang et al."}</a>{": section 3.2 explains hybrid dilation and gap conditions. Use it with explicit support enumeration rather than reading its common-factor warning as a sufficient theorem."}</li><li>{""}<a href={"https://arxiv.org/abs/1706.05587"}>{"DeepLab V3"}</a>{" and "}<a href={"https://arxiv.org/abs/1802.02611"}>{"V3+"}</a>{": respectively, parallel context branches and a boundary-refining decoder. Keep versions, output stride and training protocol distinct."}</li><li>{""}<a href={"https://d2l.ai/chapter_convolutional-neural-networks/channels.html"}>{"Dive into Deep Learning: Multiple Input and Multiple Output Channels"}</a>{": an alternate visual and code route for the local channel operations and "}<InlineMath>{"1\\times1"}</InlineMath>{" mixing. Its full standard-convolution examples are especially useful before the factorization proof; the book's own environment setup differs from our self-contained files."}</li><li>{""}<a href={"https://distill.pub/2019/computing-receptive-fields/"}>{"Distill: Computing Receptive Fields"}</a>{": an interactive geometry explanation with derivations for size, location and multi-path networks. Pair the outline diagrams with this lesson's explicit sampled-site sets."}</li><li>{""}<a href={"https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html"}>{"PyTorch Conv2d reference"}</a>{" and "}<a href={"https://docs.pytorch.org/vision/main/_modules/torchvision/models/segmentation/deeplabv3.html"}>{"Torchvision ASPP source"}</a>{": API and implementation references, not substitutes for the mechanism. The former is a stable 2.9 documentation page; the latter is a changing main-branch view inspected on 13 September 2026. Our saved calculations ran with PyTorch 2.14.0+cpu."}</li></ul>
  </div>,
};
