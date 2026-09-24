// Conserved revision-3 manuscript statically rendered at authoring time.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { LandmarkFeatureRoute, LandmarkHistoricalShapes, LandmarkKernelFigure, LandmarkBranchFigure, LandmarkInvertedRoute, LandmarkHeadLab, LandmarkContextLab, LandmarkScalingLab, LandmarkComparisonLab, LandmarkScoreLab, LandmarkRecordedMaps, LandmarkProgram, landmarkAsset } from '../../components/lesson-labs/LandmarkArchitectureLabs.jsx';
export default {
 title: 'Landmark Architectures: LeNet, AlexNet, VGG, ResNet & EfficientNet',
 readTime: '~80 min read + code, investigations and practice; optional historical branches',
 hasIntegratedGuide: true,
 content: () => <div className="neural-lesson landmark-lesson"><LessonIntro prerequisites="Convolution shapes and receptive fields, residual paths, normalization, dropout, initialization and the training loop. Each architectural mechanism and evidence boundary is refreshed where used." sections={[["1-learn-to-read-the-diagram-before-learning-the-names","1. Learn to read the diagram before learning the names"],["2-lenet-learn-local-features-and-combine-them","2. LeNet: learn local features and combine them"],["3-alexnet-and-vgg-make-richer-features-practical","3. AlexNet and VGG: make richer features practical"],["4-inception-and-resnet-change-the-routes-information-can-take","4. Inception and ResNet: change the routes information can take"],["5-efficientnet-separate-the-block-from-the-scaling-rule","5. EfficientNet: separate the block from the scaling rule"],["turn-the-architecture-diagram-into-a-complete-model","Turn the architecture diagram into a complete model"],["6-read-an-architecture-comparison-as-evidence","6. Read an architecture comparison as evidence"],["7-optional-additional-branches-in-the-architecture-family","7. Optional: additional branches in the architecture family"],["8-a-complete-small-architecture-investigation","8. A complete, small architecture investigation"],["9-how-can-a-class-score-become-a-spatial-map","9. How can a class score become a spatial map?"],["10-practice-and-diagnosis","10. Practice and diagnosis"],["11-what-you-should-now-be-able-to-do","11. What you should now be able to do"],["references-another-way-to-learn-it","References & another way to learn it"]]}>Read architectures as design decisions, build their complete compositions, then compare actual outcomes under explicit resource and data contracts.</LessonIntro>
<Prose>{""}<strong>{"Explore as you read."}</strong>{" Edit head dimensions, channel-context cells, scaling allocations, deployment budgets and signed score-map weights. Show exact parameter/MAC counts, gate contributions, candidate eligibility and current CAM/logit arithmetic live. Recorded model/seed selectors display existing evidence immediately. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to identify which operation consumes the budget, what information a head discards and why a smaller model is not automatically better."}</Prose>

<Prose>{"You can recognize a handwritten "}<strong>{"8"}</strong>{" even when one loop is wider than the other. A program receives a grid of numbers. How should its computation be arranged so that it can learn useful visual patterns, combine them, and make a decision within a resource budget?"}</Prose>

<Prose>{"An "}<strong>{"architecture"}</strong>{" is that arrangement: which operations run, the shapes they accept, and the connections through which information travels. Its weights are the numbers learned inside the arrangement. Changing the architecture changes what the model can express, how gradients reach its parameters, and what computation it requires. Changing the training recipe can also change its performance—even when the architecture stays identical."}</Prose>

<Prose>{"Building on "}<a href={"/learn/path/full-curriculum/convolution-pooling-receptive-fields?module=deep-learning-fundamentals"}>{"Convolution, Pooling & Receptive Fields"}</a>{", we will read famous networks as answers to concrete design questions. Then we will compare four small, fully specified networks on actual handwritten digits and inspect how their spatial features produce a class score."}</Prose>

<Prose>{""}<strong>{"First pass:"}</strong>{" follow §§1–6 for the architectural ideas, §8 for the runnable investigation, and §§9–11 for interpretation and practice. The detailed historical fidelity notes and §7's additional families are optional branches. You need not memorize publication years or reproduce ImageNet training to become ready for the next topic."}</Prose>

<H2>{"1. Learn to read the diagram before learning the names"}</H2>

<Prose>{"For one image, a feature tensor has shape "}<strong>{"channels × height × width"}</strong>{", written "}<code>{"C × H × W"}</code>{". A channel is a learned map of responses, not necessarily a named concept such as “eye.” With several images, the leading batch axis gives "}<code>{"N × C × H × W"}</code>{"."}</Prose>

<Prose>{"A typical classifier has three roles:"}</Prose>

<CodeBlock language={"text"}>{"pixels → stem → repeated feature-processing stages → spatial summary → class scores\n          │                  │                            │              │\n     first local maps   combine/refine maps         one vector      one number\n                                                                   per class"}</CodeBlock>

<Prose>{"The "}<strong>{"stem"}</strong>{" receives pixels. A "}<strong>{"stage"}</strong>{" usually processes maps at one spatial resolution; a boundary may reduce that resolution and increase channel count. The "}<strong>{"backbone"}</strong>{" is the feature-producing portion. A task-specific "}<strong>{"head"}</strong>{" converts its output into scores, boxes, masks, or another required answer."}</Prose>

<Prose>{"A convolution reuses the same local weight pattern across positions. An activation makes the computation nonlinear. Pooling or a strided convolution can reduce the number of positions. A final linear layer gives "}<strong>{"logits"}</strong>{", unrestricted class scores. Softmax turns those scores into a distribution; cross-entropy penalizes assigning little probability to the correct class. Backpropagation computes how every participating weight affects that loss. None of these operations knows what an “8” is before learning."}</Prose>

<Prose>{""}<strong>{"Reading the feature pyramid."}</strong>{" Our later example starts with an 8×8 input, produces twelve 8×8 maps, reduces them to twelve 4×4 maps, and eventually produces sixteen 2×2 maps and sixteen summary numbers. A grid represents positions within one channel; a stack represents separate channels. Reducing spatial size does not automatically reduce the channel axis."}</Prose>

<LandmarkFeatureRoute />

<Prose>{"Three different budgets are easily confused:"}</Prose>

<NeuralTable caption={"1. Learn to read the diagram before learning the names"} headers={[<>{"Quantity"}</>,<>{"What it counts"}</>,<>{"What changes it"}</>]} rows={[[<>{"Parameters"}</>,<>{"Stored learned scalar values"}</>,<>{"Width, kernels, connectivity, classifier dimensions"}</>],[<>{"Multiply-accumulates (MACs)"}</>,<>{"Products accumulated in convolution/linear outputs"}</>,<>{"Parameters "}<strong>{"and"}</strong>{" how many positions reuse them"}</>],[<>{"Runtime memory and elapsed time"}</>,<>{"What the actual execution needs"}</>,<>{"Shapes, batch, precision, activations, optimizer, operator implementations, device"}</>]]} />

<Prose>{"For a dense "}<code>{"k × k"}</code>{" convolution with "}<code>{"C_in"}</code>{" input and "}<code>{"C_out"}</code>{" output channels:"}</Prose>

<div className="neural-equation"><MathBlock>{"P=k^2C_{\\mathrm{in}}C_{\\mathrm{out}}+C_{\\mathrm{out}}"}</MathBlock></div>

<Prose>{"when each output channel has a bias. Omitting biases removes the final term. At "}<code>{"H_out × W_out"}</code>{" output positions, one image requires"}</Prose>

<div className="neural-equation"><MathBlock>{"M=H_{\\mathrm{out}}W_{\\mathrm{out}}k^2C_{\\mathrm{in}}C_{\\mathrm{out}}"}</MathBlock></div>

<Prose>{"convolution MACs. These counts omit bias addition, activation, normalization, pooling and other operations. Some reports count a multiply and add as two FLOPs; others report one combined operation. State the convention before comparing numbers."}</Prose>

<Prose>{"For "}<code>{"64 → 128"}</code>{", a 3×3 kernel and 14×14 output, the bias-free weight count is "}<strong>{"73,728"}</strong>{", but the convolution uses "}<strong>{"14,450,688 MACs"}</strong>{". Each learned weight is reused at 196 positions. That is why a parameter count cannot stand in for runtime."}</Prose>

<Prose>{"Before continuing, predict the effect of doubling both channel counts while leaving the spatial grid unchanged. Compare bias-free weights, convolution MACs and output-map elements."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Weights and MACs contain both channel dimensions. The output tensor contains only the output-channel dimension."}</Prose>

</details>

<details><summary>Worked reasoning</summary>

<Prose>{"The weight and convolution-MAC counts multiply by four; the number of output-map elements only doubles. These quantities have different scaling laws."}</Prose>

</details>

<H2>{"2. LeNet: learn local features and combine them"}</H2>

<Prose>{"Start with the digit task. A small patch might contain a short stroke, a curve, or background. Convolution lets one detector inspect many possible locations. A later layer combines several learned response maps, so it can respond to a configuration of earlier patterns. Subsampling reduces the spatial representation before another round of processing."}</Prose>

<Prose>{"The 1998 LeNet-5 paper describes this progression:"}</Prose>

<CodeBlock language={"text"}>{"1×32×32 → C1:6×28×28 → S2:6×14×14 → C3:16×10×10\n         → S4:16×5×5 → C5:120×1×1 → F6:84 → 10 class penalties"}</CodeBlock>

<Prose>{"The first 5×5 convolution has six filters. It needs "}<code>{"6 × (25 + 1) = 156"}</code>{" parameters, including biases. It does not learn a separate set for every patch. Its 28×28 output follows from "}<code>{"32 − 5 + 1"}</code>{" valid placements."}</Prose>

<Prose>{"Notice C5: a 5×5 convolution applied to a 5×5 map has one output location. On this input size, it behaves like a fully connected operation over that map. The operation still has spatial meaning: on a larger incoming map, the same convolution would slide over multiple locations. A shape can make two implementations coincide without making them interchangeable for every input."}</Prose>

<Prose>{"The historical model used learned subsampling coefficients, partially connected C3 channels, scaled tanh activations, and an output based on distances to class templates. Many modern “LeNet” tutorials replace these with average pooling, fully connected channel mixing and a linear classifier. Those adaptations are useful if labeled as adaptations. The architecture figure and historical details come from "}<a href={"https://gwern.net/doc/ai/nn/cnn/1998-lecun.pdf"}>{"LeCun and colleagues, §II.B"}</a>{"."}</Prose>

<LandmarkHistoricalShapes />

<Prose>{""}<strong>{"What to carry forward:"}</strong>{" the network learns a hierarchy of spatial features, and the classification loss trains that hierarchy jointly. “Early edges, later objects” can be an intuition for some learned networks; it is not a guarantee that every channel has a clean human label."}</Prose>

<Prose>{""}<strong>{"Optional historical connection."}</strong>{" The same paper connects character recognition to a larger document-reading system. A good isolated digit recognizer still needs field extraction, segmentation and contextual decisions to read a complete check. This distinction returns in modern systems: a backbone is a component, while the deployed task is an entire pipeline."}</Prose>

<H2>{"3. AlexNet and VGG: make richer features practical"}</H2>

<H3>{"AlexNet: architecture and training work together"}</H3>

<Prose>{"Recognizing varied color photographs needs more representational capacity than recognizing centered digits. AlexNet combined five convolutional and three fully connected learned layers with ReLU, augmentation, dropout and GPU training. ReLU preserves positive inputs and sets negative ones to zero; its positive-side derivative avoids the saturation of a large positive tanh input. Negative ReLU inputs still have zero derivative."}</Prose>

<Prose>{"A simplified, explicit geometry example uses a 227×227 RGB input and 96 filters of size 11, stride 4, no padding:"}</Prose>

<div className="neural-equation"><MathBlock>{"H_{\\mathrm{out}}=\\left\\lfloor\\frac{227-11}{4}\\right\\rfloor+1=55."}</MathBlock></div>

<Prose>{"A following size 3, stride 2 max pool gives 27 positions. The pool reduces the number of positions; it does not turn 96 channels into 27 channels."}</Prose>

<Prose>{""}<strong>{"One stride, then one pool."}</strong>{" One first-layer output reads an 11×11 input region. At its adjacent output, that support moves four pixels. The pool then groups 3×3 convolution responses. These are two different operations on two different grids."}</Prose>

<Prose>{"That geometry is the familiar teaching variant used in the "}<a href={"https://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf"}>{"Stanford architecture lecture"}</a>{". The original paper's input description and two-GPU connectivity, and modern library variants, need explicit matching before reproducing parameter totals. A model called “AlexNet” is not a sufficient implementation specification."}</Prose>

<Prose>{"The historical result also illustrates a measurement trap. In the original paper's ILSVRC 2012 table, one CNN has 18.2% validation top-5 error; five CNNs have 16.4%; the seven-network submission involving extra pretraining has 15.3% test error. These are different evaluation setups. Do not place 16.4 on a graph labeled “the winning single model.” "}<a href={"https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf"}>{"Krizhevsky, Sutskever and Hinton, §§3–6 and Table2"}</a>{"."}</Prose>

<H3>{"VGG: compose small filters"}</H3>

<Prose>{"Suppose every layer keeps "}<code>{"C"}</code>{" channels, stride 1 and the spatial grid. One 5×5 convolution has 25C² weights. Two 3×3 convolutions have 18C² weights, with an activation between them."}</Prose>

<Prose>{"Why do two 3×3 layers reach a 5×5 input region? The first layer reaches one pixel either side of its output center. The second reaches one first-layer position either side; each of those already depends on a 3×3 input region. The radius becomes 2. Three 3×3 layers give radius 3 and a 7×7 support."}</Prose>

<Prose>{"The same support does "}<strong>{"not"}</strong>{" mean the same function. The intermediate representation and activation create a different computation. Nor is the saving automatic when the intermediate width changes: "}<code>{"9 C_in C_mid + 9 C_mid C_out"}</code>{" must be compared with "}<code>{"25 C_in C_out"}</code>{"."}</Prose>

<LandmarkKernelFigure />

<Prose>{"VGG-16 organizes 13 convolutional layers into stages with repetition counts "}<strong>{"2,2,3,3,3"}</strong>{", channel widths "}<strong>{"64,128,256,512,512"}</strong>{", and pooling between stages. Three dense layers complete the 16 learned layers. This regular layout is easy to reason about, but its original classifier is expensive. "}<a href={"https://arxiv.org/pdf/1409.1556"}>{"VGG paper, §2 and Table1"}</a>{"."}</Prose>

<Prose>{"For the familiar 1000-class VGG-16 with biases:"}</Prose>

<NeuralTable caption={"VGG: compose small filters"} headers={[<>{"Part"}</>,<>{"Exact parameters"}</>]} rows={[[<>{"Convolutional trunk"}</>,<>{"14,714,688"}</>],[<>{""}<code>{"7×7×512 → 4096"}</code>{""}</>,<>{"102,764,544"}</>],[<>{""}<code>{"4096 → 4096"}</code>{""}</>,<>{"16,781,312"}</>],[<>{""}<code>{"4096 → 1000"}</code>{""}</>,<>{"4,097,000"}</>],[<>{"Total"}</>,<>{"138,357,544"}</>]]} />

<Prose>{"The first dense layer alone has over 102 million parameters. The "}<strong>{"whole head"}</strong>{", not that one layer, has 123,642,856. The total agrees with the "}<a href={"https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html"}>{"Torchvision VGG-16 model specification"}</a>{"."}</Prose>

<LandmarkHeadLab />

<Prose>{"The global-average alternative has "}<strong>{"513,000"}</strong>{" head parameters. It keeps one average per channel and discards within-channel spatial arrangement at this boundary. In return it avoids multiplying the head input dimension by 49. Averaging has no learned parameters; the following classifier still learns combinations of channels."}</Prose>

<Prose>{"For scale, storing VGG's first dense layer as float32 weights, gradients and two float32 Adam moment arrays requires 1,644,232,704 bytes for those four arrays. Activations, other layers and execution buffers are additional. This is an exact storage estimate under the stated representation, not a claim that a particular GPU will run out of memory."}</Prose>

<H2>{"4. Inception and ResNet: change the routes information can take"}</H2>

<H3>{"Inception: parallel views, then concatenate"}</H3>

<Prose>{"A small local pattern and a wider arrangement might both matter. An Inception-style block processes the same input along several branches and places their outputs next to one another along the channel axis:"}</Prose>

<CodeBlock language={"text"}>{"                 ┌─ 1×1 convolution ────────────────┐\ninput ───────────├─ 1×1 → activation → 3×3 ─────────┤\n                 ├─ 1×1 → activation → 5×5 ─────────┼→ concatenate channels\n                 └─ 3×3 pool → 1×1 convolution ────┘"}</CodeBlock>

<Prose>{"Every branch must return the same batch and spatial dimensions. If their channel counts are 64,128,32,32, concatenation gives 256 output channels. The operation preserves distinct branch outputs; it does not average them."}</Prose>

<Prose>{"A 1×1 convolution is learned channel mixing at each location. It can reduce width before an expensive spatial convolution. Reducing 480 channels to 32 before a 5×5 projection to 480 uses"}</Prose>

<div className="neural-equation"><MathBlock>{"480(32)+25(32)(480)=399{,}360"}</MathBlock></div>

<Prose>{"weights, instead of "}<code>{"25(480)(480)=5,760,000"}</code>{". The reduction is about 14.42-fold in this branch's weights and convolution MACs at matched output grids. It also restricts the intermediate representation. “Cheaper” is the established arithmetic; retaining enough useful information is the learning question."}</Prose>

<Prose>{"GoogLeNet combined these branches with other design and training choices. The paper acknowledges earlier work on 1×1 channel networks; it did not invent that operation in isolation. "}<a href={"https://arxiv.org/pdf/1409.4842"}>{"Szegedy and colleagues, §§4–5"}</a>{"."}</Prose>

<H3>{"ResNet: refine a representation through addition"}</H3>

<Prose>{"A residual block offers another route:"}</Prose>

<div className="neural-equation"><MathBlock>{"y=x+F(x)."}</MathBlock></div>

<Prose>{"The branch "}<code>{"F"}</code>{" learns a change to the incoming representation. If the desired transformation is close to identity, a small branch output can supply it. Backpropagation also gets a direct contribution through the identity path:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{\\partial y}{\\partial x}=I+\\frac{\\partial F}{\\partial x}."}</MathBlock></div>

<Prose>{"This helps explain the design; it does not guarantee that gradients cannot cancel or that any depth trains successfully. The preceding "}<a href={"/learn/path/full-curriculum/residual-connections-skip-connections?module=deep-learning-fundamentals"}>{"Residual Connections lesson"}</a>{" examines those limits."}</Prose>

<Prose>{"The original post-activation basic block computes two convolutions with normalization, applies ReLU between them, adds the skip, then applies ReLU again. A projection can match the skip's channels and spatial size when a stage changes shape."}</Prose>

<Prose>{""}<strong>{"Add versus concatenate."}</strong>{" For "}<code>{"x=[1,2]"}</code>{" and "}<code>{"F(x)=[3,−1]"}</code>{", addition yields "}<code>{"[4,1]"}</code>{"; concatenation yields "}<code>{"[1,2,3,−1]"}</code>{". Two aligned lanes merge at a plus sign, whereas concatenation retains four lanes side by side. The former preserves width; the latter increases it."}</Prose>

<LandmarkBranchFigure />

<Prose>{"A ResNet-50 bottleneck uses 1×1 reduction,3×3 processing, then 1×1 expansion. For 256→64→64→256, the convolutions use"}</Prose>

<div className="neural-equation"><MathBlock>{"256(64)+9(64)(64)+64(256)=69{,}632"}</MathBlock></div>

<Prose>{"weights. Two dense 3×3 convolutions at 256 channels use 1,179,648. Both return 256 channels, but their internal capacities differ. BatchNorm parameters and any skip projection must be added when counting a complete implemented block."}</Prose>

<Prose>{"ResNet's motivating "}<strong>{"degradation"}</strong>{" observation was increased "}<strong>{"training"}</strong>{" error in deeper plain networks. Calling it merely overfitting misses the optimization issue. The paper compares matched plain and residual variants; a record score also depends on its training and evaluation recipe. "}<a href={"https://arxiv.org/pdf/1512.03385"}>{"He and colleagues, introduction and §3"}</a>{"."}</Prose>

<H2>{"5. EfficientNet: separate the block from the scaling rule"}</H2>

<Prose>{"Two questions are involved: "}<strong>{"what should one block do"}</strong>{", and "}<strong>{"how should a good base network grow"}</strong>{"?"}</Prose>

<H3>{"Spatial filtering, channel mixing and a narrow skip"}</H3>

<Prose>{"A depthwise convolution filters each input channel separately. A pointwise 1×1 convolution then mixes channels. With one spatial filter per input channel, the bias-free 3×3 pair uses"}</Prose>

<div className="neural-equation"><MathBlock>{"9C_{\\mathrm{in}}+C_{\\mathrm{in}}C_{\\mathrm{out}}"}</MathBlock></div>

<Prose>{"weights. For 64→128 this is 8,768, compared with 73,728 for a dense 3×3 convolution. At 14×14 it uses 1,718,528 convolution MACs. That is about 8.41 times fewer, not a measured 8.41 times lower latency. The factorization imposes a restriction on the spatial filters that each output can use. The next topic develops that restriction explicitly. "}<a href={"https://arxiv.org/pdf/1704.04861"}>{"MobileNet V1, §3"}</a>{"."}</Prose>

<Prose>{"An "}<strong>{"inverted bottleneck"}</strong>{" first expands channels, performs depthwise spatial filtering, and projects back to a narrow representation. The skip connects the narrow representations when their shapes match:"}</Prose>

<CodeBlock language={"text"}>{"x: C channels ────────────────────────────────────────────────┐\n       └→ expand to tC → depthwise spatial → project to C ──── + → y"}</CodeBlock>

<Prose>{"This reverses the wide→narrow→wide pattern of a ResNet bottleneck. The final projection is linear in the sense that it has no subsequent pointwise activation on that branch output; the whole block is still nonlinear. A ReLU applied after a narrow projection would erase its negative coordinates. MobileNet V2 motivates preserving information there and evaluates that choice. "}<a href={"https://arxiv.org/pdf/1801.04381"}>{"MobileNet V2, §3"}</a>{"."}</Prose>

<LandmarkInvertedRoute />

<H3>{"Squeeze-and-excitation: use the image's context to gate channels"}</H3>

<Prose>{"A normal convolution has learned weights shared across input examples. An "}<strong>{"SE gate"}</strong>{" computes additional channel multipliers from the current example."}</Prose>

<Prose>{"For maps "}<code>{"U[c,h,w]"}</code>{", first average each channel:"}</Prose>

<div className="neural-equation"><MathBlock>{"s_c=\\frac{1}{HW}\\sum_{h,w}U_{c,h,w}."}</MathBlock></div>

<Prose>{"Pass the summary through a small learned network,"}</Prose>

<div className="neural-equation"><MathBlock>{"g=\\operatorname{sigmoid}\\!\\left(W_2\\operatorname{ReLU}(W_1s+b_1)+b_2\\right),\n\\qquad V_{c,h,w}=g_cU_{c,h,w}."}</MathBlock></div>

<Prose>{"The gate is constant across positions within a channel but can differ between images. Sigmoid multipliers lie between 0 and 1; they change the relative contribution of channels. A conventional convolution already mixes channels with unequal learned weights. SE adds "}<strong>{"input-dependent"}</strong>{" modulation, not the first ability to distinguish channels. "}<a href={"https://arxiv.org/pdf/1709.01507"}>{"SE paper, §3"}</a>{"."}</Prose>

<Prose>{"For a hand-sized case, take two channel means "}<code>{"a=2,b=1"}</code>{". Define one hidden unit "}<code>{"h=max(a−b,0)"}</code>{" and gate logits "}<code>{"[h,−h]"}</code>{". The gates are approximately "}<code>{"[0.7311,0.2689]"}</code>{". If the second mean becomes 3, the hidden unit becomes 0 and both gates become 0.5. Changing one channel's global content can change another channel's multiplier."}</Prose>

<LandmarkContextLab />

<H3>{"Compound scaling: spend additional resources deliberately"}</H3>

<Prose>{"EfficientNet-B0 combines mobile inverted bottlenecks with SE in a staged backbone. Its base-network search optimizes accuracy and FLOPs; the paper explicitly distinguishes this from targeting a particular device's latency. Its other central idea is to scale depth, width and input resolution together. "}<a href={"https://proceedings.mlr.press/v97/tan19a/tan19a.pdf"}>{"EfficientNet, §§3–4"}</a>{"."}</Prose>

<Prose>{"For intuition, imagine a family dominated by same-width dense convolutions. If depth is multiplied by "}<code>{"d"}</code>{", both channel dimensions by "}<code>{"w"}</code>{", and both spatial dimensions by "}<code>{"r"}</code>{", then"}</Prose>

<div className="neural-equation"><MathBlock>{"\\text{parameters}\\ \\propto dw^2,\\qquad\n\\text{convolution MACs}\\ \\propto dw^2r^2."}</MathBlock></div>

<Prose>{"Increasing resolution gives more input positions to analyze; increasing depth supplies more successive transformations; increasing width supplies more features at each stage. None alone guarantees better development performance."}</Prose>

<Prose>{"Compound scaling writes "}<code>{"d=α^φ, w=β^φ, r=γ^φ"}</code>{". Choosing "}<code>{"αβ²γ²≈2"}</code>{" makes one increment of "}<code>{"φ"}</code>{" approximately double this model's convolution work. The paper reports coefficients 1.2,1.1,1.15 from a search around B0. Their actual product is "}<strong>{"1.92027"}</strong>{", not exactly 2. At "}<code>{"φ=2"}</code>{", the idealized parameter multiplier is 2.108304 and the MAC multiplier is approximately 3.687437."}</Prose>

<Prose>{"Do not identify every named B-index with that integer "}<code>{"φ"}</code>{" and then present the approximation as an exact model count. Implementations round channels and repeat counts; depthwise, pointwise, SE, stem and classifier terms do not all scale with the same exponents. Input resolution changes MACs without directly changing stored convolution weights."}</Prose>

<LandmarkScalingLab />

<H2>{"Turn the architecture diagram into a complete model"}</H2>

<Prose>{"The small digit comparison later in this page isolates routing decisions. To also build the named families, "}<a href={"/learn-assets/landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet/landmark_builders.py"}>{"landmark_builders.py"}</a>{" supplies complete model constructors from ordinary "}<code>{"nn.Conv2d"}</code>{", normalization, activation, pooling and linear primitives. Their internal operations have already been opened in the preceding convolution, normalization, residual and dropout lessons. Here the new mechanism is "}<strong>{"composition"}</strong>{": stage widths, repetition counts, downsampling locations, parallel channel gating and the final head."}</Prose>

<NeuralTable caption={"Turn the architecture diagram into a complete model"} headers={[<>{"Builder"}</>,<>{"Exact declared construction"}</>]} rows={[[<>{""}<code>{"lenet"}</code>{""}</>,<>{"32×32 single-channel dense-convolution/tanh/average-pool model with 6 and16 maps, followed by120→84→class head; the modern variant contrasted with original LeNet-5 above"}</>],[<>{""}<code>{"alexnet"}</code>{""}</>,<>{"Torchvision-style 64/192/384/256/256 feature widths, 11/5/3/3/3 kernels and 6×6 adaptive head grid; not the historical two-device grouping/LRN recipe"}</>],[<>{""}<code>{"vgg16"}</code>{""}</>,<>{"2/2/3/3/3 convolution repetitions and the complete 7×7×512→4096→4096→class head"}</>],[<>{""}<code>{"resnet18"}</code>{""}</>,<>{"Explicit two-convolution post-activation blocks; stride/projection on each changing stage; global pooling and class head"}</>],[<>{""}<code>{"efficientnet_b0"}</code>{""}</>,<>{"All seven B0 stage configurations, expansion/depthwise/SE/projection, per-example branch dropping and1280-channel head"}</>]]} />

<Prose>{"Every builder returns a trainable "}<code>{"nn.Module"}</code>{", not a call to a hidden model factory. Read "}<code>{"ResidualBlock"}</code>{" and "}<code>{"MobileBlock"}</code>{" next to their diagrams. The B0 squeeze width is based on the block's "}<strong>{"incoming"}</strong>{" width, rather than blindly reducing the expanded tensor by four. Its SE activation is SiLU. Those are concrete differences from the earlier deliberately smaller illustrative gate. Keeping both examples is useful because the comparison now identifies their different contracts instead of calling them the same model."}</Prose>

<Prose>{""}<code>{"python landmark_builders.py --family vgg16"}</code>{" uses "}<strong>{"meta tensors"}</strong>{" to inspect the full shape and parameter count without allocating138 million weights. It should report the specified parameter budget and "}<code>{"[1,1000]"}</code>{" output; this checks a construction, not a fit. Changing the classifier requires its output label count to match the task, and does not preserve pretrained category semantics."}</Prose>

<Prose>With <code>torchvision==0.29.0</code> paired with <code>torch==2.14.0</code>, run <code>python landmark_builders.py --family resnet18 --compare</code>. It constructs the ordinary <code>get_model(..., weights=None)</code> route too. <code>copy_components</code> copies every convolution, linear layer and BatchNorm state in semantic order, refusing a component count/type/shape mismatch. Both models are in evaluation mode and receive the same input. Actual separate CPU comparisons for ResNet-18, EfficientNet-B0, VGG-16 and AlexNet each produced maximum absolute logit difference 0 in the recorded environment. This compares matched random weights, not trained quality. The VGG/AlexNet commands allocate both full models; run those families individually when memory permits, never in the browser.</Prose>

<Prose>{"The model source is intentionally a readable composition, not a reproduction of historical hardware kernels or each paper's training recipe. A Torchvision update can change a component layout: the comparison should fail visibly so the mapping can be inspected. "}<a href={"https://docs.pytorch.org/vision/stable/models.html"}>{"The model API"}</a>{" and "}<a href={"https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/efficientnet.py"}>{"EfficientNet implementation"}</a>{" were checked for this bridge on22September2026. Pin the compatible tested release when running it."}</Prose>

<LandmarkProgram file="landmark_builders.py" title="Read all five explicit architecture builders and the library comparison" /><Prose>The reusable mobile block accepts branch-drop probability from 0 through 1. At 1 it keeps only the identity path during training without dividing by zero; out-of-range values are rejected. Default B0 probabilities remain below 1. <a href={landmarkAsset + "native-verification.json"}>Executed construction, boundary and environment checks</a>.</Prose>

<Prose>{""}<strong>{"Use the ordinary pretrained route."}</strong>{" Select an explicit weight enum, obtain its input transform, run the matching model in eval/inference mode and interpret outputs using that enum's categories. The following complete example deliberately requires a local photograph; it downloads model weights if uncached:"}</Prose>

<CodeBlock language={"python"}>{"import torch\nfrom PIL import Image\nfrom torchvision.models import ResNet18_Weights, resnet18\n\nweights = ResNet18_Weights.IMAGENET1K_V1\nmodel = resnet18(weights=weights).eval()\nwith Image.open(\"example.jpg\") as image:\n    inputs = weights.transforms()(image.convert(\"RGB\")).unsqueeze(0)\nwith torch.inference_mode():\n    probabilities = model(inputs).softmax(-1)[0]\nfor index in probabilities.topk(5).indices:\n    print(weights.meta[\"categories\"][int(index)], float(probabilities[index]))"}</CodeBlock>

<Prose>{"This does not turn a random local model into pretrained ResNet by assigning the same name. For fine-tuning, reuse the already implemented "}<a href={"/learn/path/full-curriculum/transfer-learning-fine-tuning-strategies#transfer-section-3"}>{"Transfer Learning pipeline, section3"}</a>{", whose "}<code>{"transfer-experiments.py"}</code>{" owns split roles, changed heads, freeze policies and optimizer groups. Apply those policies to the chosen architecture; do not rerun a second transfer lesson here."}</Prose>

<Prose>The photograph/IMAGENET1K_V1 download example is a complete usage route, but it was not executed here: no external photograph or pretrained weights were downloaded. The offline matched-state construction comparisons above were executed. To reproduce this environment, create and activate a separate Python environment, then install <code>torch==2.14.0 torchvision==0.29.0 numpy==2.3.5 scikit-learn==1.9.1 pillow</code> with <code>python -m pip install</code>.</Prose>

<Prose>{""}<strong>{"Construction exercise."}</strong>{" Replace VGG's original head with global averaging and a seven-class linear layer. Which code and tensor contracts change?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"The trunk still returns512 channels; the head no longer receives49 positions per channel."}</Prose>

</details>

<details>

<summary>Solution and success criteria</summary>

<Prose>{"Keep the feature stages, set the pool to "}<code>{"nn.AdaptiveAvgPool2d(1)"}</code>{", and use "}<code>{"nn.Linear(512,7)"}</code>{". Its3,591 head parameters replace the three large dense layers. Test two legal image sizes and assert "}<code>{"[batch,7]"}</code>{"; train using labels0–6. The unchanged trunk can receive copied pretrained state, but the replaced head must learn its new label meanings. Successful shape and parameter checks do not prove equal accuracy or equal functions."}</Prose>

</details>

<H2>{"6. Read an architecture comparison as evidence"}</H2>

<Prose>{"A useful comparison tells you "}<strong>{"which model, which weights, which data and split, which preprocessing, which metric, and which execution conditions"}</strong>{"."}</Prose>

<Prose>{"Top-1 accuracy asks whether the largest score selects the label. Top-5 accuracy asks whether the label appears among the five largest scores. Error is one minus accuracy. A multi-network ensemble is a different deployed computation from a single model. A model trained with extra data or a newer recipe is a different experiment even if its diagram looks familiar."}</Prose>

<Prose>{"For example, Torchvision documents "}<strong>{"76.130%"}</strong>{" ImageNet-1K top-1 for ResNet-50's "}<code>{"IMAGENET1K_V1"}</code>{" weights and "}<strong>{"80.858%"}</strong>{" for "}<code>{"IMAGENET1K_V2"}</code>{". The architecture and 25,557,032 parameter count are the same. The difference is evidence that the trained-weight package and recipe matter, not a new residual-connection discovery. The V2 inference transform resizes to 232 before a 224 center crop; it is not simply “every image model takes 224.” "}<a href={"https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html"}>{"ResNet-50 weights and transforms"}</a>{"."}</Prose>

<Prose>{"Use this separation when selecting candidates:"}</Prose>

<NeuralTable caption={"6. Read an architecture comparison as evidence"} headers={[<>{"Deployment question"}</>,<>{"Evidence to obtain"}</>]} rows={[[<>{"Can this recognize the target classes?"}</>,<>{"Task-specific development metric, baseline, error examples and important slices"}</>],[<>{"Can this preserve small details?"}</>,<>{"Input resolution and stage geometry; inspect what preprocessing discards"}</>],[<>{"Does it fit the device?"}</>,<>{"Export/operator support, measured latency at the intended batch and precision, peak memory"}</>],[<>{"Can it reuse a pretrained model?"}</>,<>{"Exact weight identifier, preprocessing, class/head contract and appropriate adaptation"}</>],[<>{"Will its features feed another task?"}</>,<>{"Required stage resolutions, channel dimensions and output meanings"}</>]]} />

<Prose>{"A "}<strong>{"Pareto"}</strong>{" comparison concerns competing objectives: a candidate is dominated if another is at least as good on all declared objectives and strictly better on one. You cannot declare one architecture family universally Pareto-optimal from unrelated runs. A faster model on one CPU may be slower on another accelerator because operator efficiency and memory traffic change."}</Prose>

<Prose>{"Keep fine-tuning decisions with the preceding "}<a href={"/learn/path/full-curriculum/transfer-learning-fine-tuning-strategies?module=deep-learning-fundamentals"}>{"Transfer Learning lesson"}</a>{": freezing parameters and choosing BatchNorm behavior are separate decisions. A small target dataset does not establish a universal transfer gain or require full fine-tuning in every case."}</Prose>

<H2>{"7. Optional: additional branches in the architecture family"}</H2>

<Prose>{"These ideas complete the useful historical context without making every named family a prerequisite for the next lesson."}</Prose>

<Prose>{""}<strong>{"DenseNet retains earlier maps by concatenation."}</strong>{" A layer receives "}<code>{"[x₀,x₁,…,xₗ₋₁]"}</code>{" and produces a small set of new channels. Starting with 8 channels and adding 3 per layer gives widths 8,11,14,17,20. Reusing prior maps can support feature and gradient access, while the widening inputs and retained activations affect computation and memory. Transition layers can compress channels and downsample. This is a different connectivity contract from residual addition. "}<a href={"https://arxiv.org/pdf/1608.06993"}>{"DenseNet, §3"}</a>{"."}</Prose>

<Prose>{""}<strong>{"RegNet asks about a family of designs."}</strong>{" Start with proposed block widths "}<code>{"u_j=w₀+w_a j"}</code>{", quantize them into repeated widths, and group consecutive equal-width blocks into stages. The result is a small set of design parameters controlling an entire network. Evaluating distributions of sampled designs asks whether a design space reliably produces good candidates, rather than celebrating one searched winner. Its empirical conclusions depend on its search and evaluation protocol. "}<a href={"https://arxiv.org/pdf/2003.13678"}>{"RegNet, §3"}</a>{"."}</Prose>

<Prose>{""}<strong>{"NFNet separates normalization from the requirements it helps satisfy."}</strong>{" Its construction combines scaled weight standardization, controlled residual-branch scales and adaptive gradient clipping. The clipping threshold depends on a gradient norm relative to a parameter norm; it is not the same operation as multiplying a residual branch by a constant. This illustrates a general lesson: removing BatchNorm responsibly requires addressing training behavior, not simply deleting a module and expecting the old recipe to work. "}<a href={"https://arxiv.org/pdf/2102.06171"}>{"NFNet, §§3–4"}</a>{"."}</Prose>

<Prose>{""}<strong>{"Learned features can define another model's loss."}</strong>{" In perceptual-loss work, an image transformation network produces an image "}<code>{"ŷ"}</code>{". A separately pretrained, frozen feature network "}<code>{"φ"}</code>{" maps "}<code>{"ŷ"}</code>{" and a target image "}<code>{"y"}</code>{" into features. A loss such as"}</Prose>

<div className="neural-equation"><MathBlock>{"L_{\\mathrm{feature}}=\\frac{1}{CHW}\\|\\phi_j(\\hat y)-\\phi_j(y)\\|_2^2"}</MathBlock></div>

<Prose>{"compares one layer's representation. Gradients pass through the frozen feature computation to the generated image, even though the feature network's weights are not being updated. A deeper layer can tolerate pixel changes that a pixelwise loss heavily penalizes, but the chosen features can also overlook changes that matter to a human. This is a useful application of VGG's intermediate maps, not a guarantee of perceptual correctness. "}<a href={"https://arxiv.org/pdf/1603.08155"}>{"Johnson, Alahi and Fei-Fei, §3.2"}</a>{"."}</Prose>

<Prose>{"MobileNet refinements and ConvNeXt follow in their own lessons. Wide residual networks change channel capacity; grouped ResNeXt branches change the transformation grouping. Stochastic-depth training already has a "}<a href={"/learn/path/full-curriculum/dropout-droppath-stochastic-depth?module=deep-learning-fundamentals"}>{"separate home"}</a>{". These are combinations of design choices, not steps on a ladder where every later name makes every earlier one obsolete."}</Prose>

<H2>{"8. A complete, small architecture investigation"}</H2>

<H3>{"The question and the observations"}</H3>

<Prose>{"Can different feature-processing blocks learn to recognize actual digits, and what do they cost? Use 400 real 8×8 images from the "}<strong>{"UCI Optical Recognition of Handwritten Digits"}</strong>{" data: 40 images of each digit 0–9. The original collection supports studying handwritten-digit recognition. Each pixel is an integer intensity 0–16; this dataset is different from MNIST. The supplied subset, source-row identifiers and attribution are in "}<a href={"/learn-assets/landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet/digits-400.csv"}>{"digits-400.csv"}</a>{" and "}<a href={"/learn-assets/landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet/data-provenance.md"}>{"data provenance"}</a>{". The UCI record distributes this dataset under CC BY 4.0. "}<a href={"https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits"}>{"Dataset and collectors"}</a>{"."}</Prose>

<Prose>{"All 400 source IDs and pixel vectors are distinct. We use a stratified 280/120 training/development split with seed 22 and divide pixels by the known bound 16. No fitted preprocessing consumes development rows. The source copy does not provide writer identities for these rows, so this does not establish performance on unseen writers. Development results are available for inspection and model choice; there is no untouched final test in this teaching experiment."}</Prose>

<Prose>{"Every candidate has the same external shape route:"}</Prose>

<CodeBlock language={"text"}>{"N×1×8×8\n  → 3×3 convolution,1→12; ReLU; max pool2\nN×12×4×4\n  → one selected body\nN×12×4×4\n  → 3×3 convolution,12→16; ReLU; max pool2\nN×16×2×2\n  → average each map\nN×16\n  → linear16→10\nN×10 logits"}</CodeBlock>

<Prose>{"The four bodies are small teaching constructions:"}</Prose>

<NeuralTable caption={"The question and the observations"} headers={[<>{"Body"}</>,<>{"Operation on 12×4×4 input"}</>]} rows={[[<>{"Plain"}</>,<>{"3×3→ReLU→3×3→ReLU, all 12 channels"}</>],[<>{"Residual"}</>,<>{"The same two convolutions, add the input before the final ReLU"}</>],[<>{"Parallel"}</>,<>{"Four branches returning 3 channels each:1×1;1×1→3×3;1×1→5×5;pool→1×1; concatenate and apply ReLU"}</>],[<>{"Inverted with gate"}</>,<>{"12→36 expansion; depthwise 3×3; SE hidden width 9;36→12 linear projection; add input"}</>]]} />

<Prose>{"These are "}<strong>{"not miniature benchmark reproductions"}</strong>{" of VGG, GoogLeNet, ResNet or EfficientNet. We omit BatchNorm and use fixed small stages so the bodies are readable. The inverted block uses SiLU, the smooth activation "}<code>{"x × sigmoid(x)"}</code>{", in expansion/spatial operations; the others use ReLU. The comparisons between different body families change several properties at once."}</Prose>

<Prose>{"Within a seed, stem, tail and head start with matching tensors in all four models. Plain and residual also share the same initial branch weights: their only forward-rule difference is the skip addition. They are trained separately, so their weights can subsequently diverge."}</Prose>

<H3>{"Run it offline and follow one update"}</H3>

<Prose>{"Download "}<a href={"/learn-assets/landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet/architecture-experiments.py"}>{"architecture-experiments.py"}</a>{" beside the CSV. In a Python environment with PyTorch, NumPy and scikit-learn installed, run:"}</Prose>

<CodeBlock language={"text"}>{"python architecture-experiments.py"}</CodeBlock>

<Prose>{"The complete program defines every body, loads the supplied data, checks the split inputs, runs 12 small fits, prints final numerical summaries, and writes "}<code>{"calculated-inputs.json"}</code>{". It needs no pretrained download or image folder. The recorded run used Python 3.12.14, PyTorch 2.14.0+cpu, NumPy 2.3.5 and scikit-learn 1.9.1 with one CPU thread. Exact low-order values can vary with numerical libraries."}</Prose>

<LandmarkProgram /><p><a href={landmarkAsset + "calculated-inputs.json"}>Complete measured experiment record</a> · <a href={landmarkAsset + "native-verification.json"}>Execution evidence and limits</a></p>

<Prose>{"Read its two-convolution body first. The complete program supplies "}<code>{"torch"}</code>{", "}<code>{"nn"}</code>{" and "}<code>{"F"}</code>{" imports and calls this inside the classifier:"}</Prose>

<CodeBlock language={"python"}>{"class PlainOrResidual(nn.Module):\n    def __init__(self, channels=12, residual=False):\n        super().__init__()\n        self.first = nn.Conv2d(channels, channels, 3, padding=1)\n        self.second = nn.Conv2d(channels, channels, 3, padding=1)\n        self.residual = residual\n\n    def forward(self, x):\n        branch = self.second(F.relu(self.first(x)))\n        return F.relu(branch + x if self.residual else branch)"}</CodeBlock>

<Prose>{"Padding 1 preserves the 4×4 grid. Both convolutions preserve 12 channels, so addition is shape-compatible. The last ReLU belongs after the merge in this example. Moving it into only the branch changes the function."}</Prose>

<Prose>{"The training mechanism is compact:"}</Prose>

<CodeBlock language={"python"}>{"optimizer = torch.optim.Adam(model.parameters(), lr=.003)\nfor step in range(400):\n    model.train()\n    optimizer.zero_grad()\n    loss = F.cross_entropy(model(x[train]), y[train])\n    loss.backward()\n    optimizer.step()"}</CodeBlock>

<Prose>{"This excerpt is the update loop from the complete supplied program, not a standalone script. Each step uses all 280 training images. The logits select neither labels nor probabilities in advance: cross-entropy performs the needed log-softmax internally. Clearing gradients prevents the previous step's gradients from being added accidentally. Backpropagation reaches the head, tail, selected body and stem. Adam then updates their learned parameters."}</Prose>

<Prose>{"There are no augmentations, dropout, weight decay, early stopping or development-selected epochs. Every model receives 400 updates at learning rate 0.003 with seeds 1,2,3. We record training/development cross-entropy and correct counts at steps 0,1,25,100,200,400. This fixed recipe is a comparison condition, not a promise that it is optimal for all four architectures."}</Prose>

<H3>{"Inspect the result, including the baseline"}</H3>

<Prose>{"All 12 final models correctly classify all 280 training images. Their development behavior differs:"}</Prose>

<NeuralTable caption={"Inspect the result, including the baseline"} headers={[<>{"Body"}</>,<>{"Parameters"}</>,<>{"Conv/linear MACs per image"}</>,<>{"Development correct, seeds 1/2/3, out of 120"}</>,<>{"Development CE, seeds 1/2/3"}</>]} rows={[[<>{"Plain"}</>,<>{"4,650"}</>,<>{"76,192"}</>,<>{"116 /115 /112"}</>,<>{"0.1180 /0.2325 /0.3061"}</>],[<>{"Residual"}</>,<>{"4,650"}</>,<>{"76,192"}</>,<>{"112 /115 /114"}</>,<>{"0.2065 /0.2074 /0.3737"}</>],[<>{"Parallel"}</>,<>{"2,502"}</>,<>{"41,920"}</>,<>{"119 /117 /115"}</>,<>{"0.0367 /0.1056 /0.2091"}</>],[<>{"Inverted with gate"}</>,<>{"3,999"}</>,<>{"54,376"}</>,<>{"116 /112 /117"}</>,<>{"0.1971 /0.3331 /0.0928"}</>]]} />

<Prose>{"The MAC count excludes addition, pooling, gate multiplication and activations. Thus plain and residual have equal counted MACs, although residual addition still performs work. The saved layer-by-layer counts make this convention inspectable."}</Prose>

<Prose>{"What can we conclude? All four small constructions learn the training set. The parallel candidate has fewer counted parameters and MACs here, and its three development counts are promising. The residual path does not consistently beat the plain model in this shallow fixed-recipe setting. A result about helping train very deep networks is not a guarantee that a skip improves every small model."}</Prose>

<Prose>{"Notice the inverted candidate's seed 1 score: 116 correct, the same as plain, but higher cross-entropy. Correct counts discard confidence information. Cross-entropy also reacts to probability assigned to wrong labels and to the confidence of correct predictions."}</Prose>

<LandmarkComparisonLab />

<Prose>{"The three seeds show initialization variation on one shared split. They do not provide independent samples from a deployment population. If you pick the parallel candidate after inspecting this table, that choice has consumed development information; a later final assessment needs new held-out evidence."}</Prose>

<H2>{"9. How can a class score become a spatial map?"}</H2>

<Prose>{"The final representation in our classifier is 16 maps of size 2×2. The head averages each map, then linearly combines those 16 averages. Because averaging and a weighted sum are linear, we can reverse their order."}</Prose>

<Prose>{"Let "}<code>{"A_c(h,w)"}</code>{" be the final map in channel "}<code>{"c"}</code>{", and let "}<code>{"w_kc,b_k"}</code>{" be the weights and bias for class "}<code>{"k"}</code>{":"}</Prose>

<div className="neural-equation"><MathBlock>{"z_k=b_k+\\sum_c w_{kc}\\left(\\frac{1}{HW}\\sum_{h,w}A_c(h,w)\\right)."}</MathBlock></div>

<Prose>{"Define the "}<strong>{"class activation map"}</strong>{""}</Prose>

<div className="neural-equation"><MathBlock>{"M_k(h,w)=\\sum_c w_{kc}A_c(h,w)."}</MathBlock></div>

<Prose>{"Then "}<code>{"z_k=b_k+mean(M_k)"}</code>{". The mean of the map, plus the bias, recovers the class logit exactly in real arithmetic for this head. A map is not itself a probability map, and a negative contribution can reduce the class score. This correspondence is the mechanism behind "}<a href={"https://arxiv.org/pdf/1512.04150"}>{"class activation mapping"}</a>{"; our definition explicitly uses an average and keeps the bias."}</Prose>

<Prose>{"Here is a complete hand-sized computation you can run independently:"}</Prose>

<CodeBlock language={"python"}>{"import numpy as np\n\nmaps = np.array([[[1., 2.], [0., 3.]],\n                 [[0., 1.], [2., 1.]]])\nweights = np.array([2., -1.])\nbias = .5\nclass_map = np.einsum(\"c,chw->hw\", weights, maps)\nvia_features = weights @ maps.mean(axis=(1, 2)) + bias\nvia_map = class_map.mean() + bias\nprint(class_map)\nprint(via_features, via_map)"}</CodeBlock>

<Prose>{"Output:"}</Prose>

<CodeBlock language={"text"}>{"[[ 2.  3.]\n [-2.  5.]]\n2.5 2.5"}</CodeBlock>

<Prose>{"The second channel has a negative class weight. Raising its bottom-left cell from 2 to 6 changes that location's class-map value from −2 to −6 and lowers the score from 2.5 to 1.5. More activation can mean less evidence for this particular class."}</Prose>

<LandmarkScoreLab />

<Prose>{"Then inspect real saved examples from seed 1. Source 251, an actual 4, is correctly classified by all four models. The first misclassified development specimen for the parallel model is source 379: an 8 predicted as 5. Its class 5 and class 8 maps come from the "}<strong>{"same"}</strong>{" feature tensor with different head weights. Looking only at a vivid predicted-class map would hide the competing explanation."}</Prose>

<Prose>{"The saved features, all ten class maps, head weights, biases, logits and probabilities support the correspondence. Float32 reconstructions differ by at most approximately 5.73×10⁻⁶ across the saved cases. That is numerical ordering error in an algebraic identity."}</Prose>

<LandmarkRecordedMaps />

<Prose>{"Use the original 2×2 cells alongside any enlarged overlay. Upsampling does not create extra spatial detail. A large positive cell describes a contribution from learned features whose receptive field can extend beyond that cell's displayed location. It does not prove a causal explanation, a precise object boundary, or that editing the corresponding input pixels will have the predicted effect. Editing intermediate features in the hand calculation is explicitly a different intervention from rerunning an image through the whole trained backbone."}</Prose>

<H2>{"10. Practice and diagnosis"}</H2>

<Prose>{"Attempt each task before opening its hint or solution."}</Prose>

<H3>{"1. A new head budget"}</H3>

<Prose>{"A backbone returns 128 channels of size 7×7. Compare a direct flattened linear head for 7 classes with global averaging followed by a linear head. Include biases. Which spatial information can only the first head use?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"One head receives 6,272 scalars; the other receives 128. Both return 7 logits."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Flattened: "}<code>{"(7×7×128+1)×7=43,911"}</code>{" parameters. Averaged: "}<code>{"(128+1)×7=903"}</code>{". The flattened head can assign different weights to different positions within a channel. The averaged head cannot distinguish permutations of those positions at its input. This does not establish which trained model has better task performance."}</Prose>

</details>

<H3>{"2. A bottleneck that is no longer cheap"}</H3>

<Prose>{"Compare one 5×5 convolution from 32 to 32 channels with two 3×3 convolutions, first 32→64 then 64→32. Ignore biases, use the same spatial grid, and put an activation between the small convolutions. Is “two small kernels use fewer weights” true here?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Write both channel dimensions of both 3×3 layers. The intermediate width is 64, not 32."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The 5×5 layer has "}<code>{"25×32²=25,600"}</code>{" weights. The pair has "}<code>{"9×32×64+9×64×32=36,864"}</code>{". Their interior receptive-field support is 5×5, but the pair is larger and has an intermediate nonlinearity. The familiar 18C² comparison assumes the intermediate width is C."}</Prose>

</details>

<H3>{"3. Repair the merge"}</H3>

<Prose>{"An input has shape "}<code>{"N×24×16×16"}</code>{". A strided branch returns "}<code>{"N×48×8×8"}</code>{". Give a shape-compatible learned skip for addition. If another design concatenates two "}<code>{"N×24×8×8"}</code>{" branches, what is its output shape? Explain why these are different computations despite one matching final shape."}</Prose>

<details><summary>Hint</summary>

<Prose>{"An addition needs both summands to match. A 1×1 convolution can change channels and use a stride."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"A 1×1 skip convolution 24→48 with stride 2 returns "}<code>{"N×48×8×8"}</code>{". The branch and projected skip can then be added. Concatenating two 24-channel branches also returns "}<code>{"N×48×8×8"}</code>{", but retains their outputs in separate channel ranges. Addition combines corresponding entries. With no bias, the projection has 24×48=1,152 weights; its 8×8 outputs require 73,728 convolution MACs per image."}</Prose>

</details>

<H3>{"4. Change the class-map question"}</H3>

<Prose>{"Use the two maps in §9 with new class weights "}<code>{"[−1,2]"}</code>{" and bias −0.5. Compute the map and score. Increase the first map's top-left cell from 1 to 5. Predict, then calculate, the new score."}</Prose>

<details><summary>Hint</summary>

<Prose>{"The first map now has negative weight. A change at one of four positions changes the spatial average by one quarter of the weighted change."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The original class map is "}<code>{"[[-1,0],[4,-1]]"}</code>{". Its mean is 0.5, so the score is 0. The edit reduces the top-left contribution by 4; the map's mean falls by 1 and the score becomes −1. The other class from §9 can respond differently to exactly the same features."}</Prose>

</details>

<H3>{"5. Can you draw this benchmark curve?"}</H3>

<Prose>{"A draft has a point for an old model's single-crop validation accuracy, another for a later seven-network test ensemble with extra pretraining, and invented points filling missing years. Its caption says “architecture progress.” Describe a defensible replacement."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Separate what is known, what is comparable, and what was not measured."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Remove invented points. Either select a genuinely matched protocol or show discrete reported results with explicit weights, data, split, inference and source labels. A historical table can preserve the milestones without implying a controlled causal comparison. For this lesson's architecture budgets, use exact shape/parameter calculations separately from the recorded small-data results. Connecting points is an additional claim about what the line means."}</Prose>

</details>

<H3>{"6. Diagnose a failed adaptation"}</H3>

<Prose>{"A team freezes all parameters whose name lacks the text "}<code>{"\"classifier\""}</code>{" or "}<code>{"\"fc\""}</code>{". Its supposedly frozen backbone contains SE layers named "}<code>{"fc1"}</code>{" and "}<code>{"fc2"}</code>{". It also calls "}<code>{"model.train()"}</code>{" globally. Why might this fail to implement a linear probe? Propose direct checks."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Module identity is stronger than a substring. Learned tensors and running-state buffers need separate inspection."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The substring rule can leave backbone SE weights trainable. Global train mode can update BatchNorm running statistics even for parameters with "}<code>{"requires_grad=False"}</code>{". Freeze the backbone module's actual parameters, explicitly make only the intended head trainable, and set the chosen backbone evaluation policy after any global mode change. List optimizer parameter identities and compare backbone parameters/buffers before and after a step. A chosen fine-tuning policy may deliberately update some of these; label that policy accurately."}</Prose>

</details>

<H3>{"7. Design a next experiment"}</H3>

<Prose>{"You have a 5,000-parameter limit and a 60,000-convolution/linear-MAC limit for the small task. Which recorded candidates qualify? Choose one development question to investigate next and state what would remain unknown."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Eligibility is arithmetic. Choosing a model and estimating its final deployment performance require different evidence."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Parallel and inverted-with-gate qualify; plain/residual exceed the MAC limit. One reasonable next question is whether the parallel model's errors concentrate in a particular pair of digits, inspected on development rows with denominators. Another is measured device latency, since counted MACs omit important work. Either investigation consumes development or engineering evidence. Unseen-writer reliability and final selected-model performance remain unestablished. More than one next experiment can be sensible if its question and decision rule are explicit."}</Prose>

</details>

<H2>{"11. What you should now be able to do"}</H2>

<Prose>{"Explain a network as a flow of tensors, not a list of names. Predict the shape and budget consequences of a new layer. Distinguish stacking small kernels, concatenating branches, adding a residual, gating channels and scaling a family. Follow data through a complete training/evaluation example and explain why a compelling architectural idea can still lose a particular comparison. Reconstruct a class score from its maps and identify the interpretation's limits."}</Prose>

<Prose>{"You are ready to continue when you can repair the mismatched merge in practice 3, derive the changed map in practice 4, and propose a defensible comparison in practice 7 without copying an architecture recommendation."}</Prose>

<Prose>{"The next topic is "}<a href={"/learn/path/full-curriculum/depthwise-separable-dilated-convolutions?module=deep-learning-fundamentals"}>{"Depthwise Separable & Dilated Convolutions"}</a>{". We have used a factorized convolution as a building block; next we examine exactly which channel/spatial interactions it can express, how dilation changes the positions a filter samples, and when those choices help or fail."}</Prose>

<H2>{"References & another way to learn it"}</H2>

<Prose>{""}<strong>{"Alternate explanations and practice"}</strong>{""}</Prose>

<ul><li>{""}<a href={"https://www.youtube.com/watch?v=DAOcjicFr1Y"}>{"Stanford CS231n, Lecture9: CNN Architectures"}</a>{", video, with "}<a href={"https://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf"}>{"companion slides"}</a>{". Useful after §3 for shape questions and visual comparisons of AlexNet, VGG, GoogLeNet and ResNet. It assumes basic convolution and predates EfficientNet. The relevant slide content and the creator's video description were reviewed; no claim of watching the complete video or verifying timestamps is made. Historical variants and current APIs still need the distinctions in this lesson."}</li><li>{""}<a href={"https://docs.pytorch.org/vision/stable/models.html"}>{"Torchvision's model and weight guide"}</a>{", official documentation. Use after §6 to connect a model builder with its weight identifier, transformations and output categories. It is an API guide rather than a beginner explanation of the architecture."}</li><li>{""}<a href={"https://arxiv.org/pdf/1603.08155"}>{"Perceptual Losses for Real-Time Style Transfer and Super-Resolution"}</a>{", Johnson, Alahi and Fei-Fei, paper. Optional after §7:§3.2 and its reconstruction figures make intermediate-feature losses concrete. Basic backpropagation is useful; its historical feature loss is not a universal human-similarity metric."}</li></ul>

<Prose>{""}<strong>{"Precise technical and historical sources"}</strong>{""}</Prose>

<ul><li>{""}<a href={"https://gwern.net/doc/ai/nn/cnn/1998-lecun.pdf"}>{"LeNet and document recognition"}</a>{", LeCun and colleagues, 1998, original paper mirrored as a PDF; §II.B for the actual layer contract."}</li><li>{""}<a href={"https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf"}>{"AlexNet"}</a>{", 2012, §§3–6 and Table 2; "}<a href={"https://arxiv.org/pdf/1409.1556"}>{"VGG"}</a>{", §2 and Table 1; "}<a href={"https://arxiv.org/pdf/1409.4842"}>{"GoogLeNet/Inception"}</a>{", §§4–5; "}<a href={"https://arxiv.org/pdf/1512.03385"}>{"ResNet"}</a>{", §3. Read a model's experimental conditions together with its diagram."}</li><li>{""}<a href={"https://arxiv.org/pdf/1704.04861"}>{"MobileNet V1"}</a>{", §3; "}<a href={"https://arxiv.org/pdf/1801.04381"}>{"MobileNet V2"}</a>{", §3; "}<a href={"https://arxiv.org/pdf/1709.01507"}>{"SE"}</a>{", §3; "}<a href={"https://proceedings.mlr.press/v97/tan19a/tan19a.pdf"}>{"EfficientNet"}</a>{", §§3–4. These supply the efficient-block and scaling definitions."}</li><li>{""}<a href={"https://arxiv.org/pdf/1608.06993"}>{"DenseNet"}</a>{", §3; "}<a href={"https://arxiv.org/pdf/2003.13678"}>{"RegNet"}</a>{", §3; "}<a href={"https://arxiv.org/pdf/2102.06171"}>{"NFNet"}</a>{", §§3–4. Optional family branches rather than required extra reading."}</li><li>{""}<a href={"https://arxiv.org/pdf/1512.04150"}>{"Class activation mapping"}</a>{", Zhou and colleagues, §2. Compare its pooled-feature convention with the explicit mean and bias kept here."}</li><li>{""}<a href={"https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits"}>{"UCI optical digits"}</a>{", E. Alpaydin and C. Kaynak, 1998, "}<a href={"https://doi.org/10.24432/C50P49"}>{"DOI10.24432/C50P49"}</a>{", CC BY 4.0; "}<a href={"/learn-assets/landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet/architecture-experiments.py"}>{"complete program"}</a>{", "}<a href={"/learn-assets/landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet/calculated-inputs.json"}>{"retained numerical outputs"}</a>{", "}<a href={"/learn-assets/landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet/data-provenance.md"}>{"data provenance and limits"}</a>{"."}</li></ul>
</div>
};
