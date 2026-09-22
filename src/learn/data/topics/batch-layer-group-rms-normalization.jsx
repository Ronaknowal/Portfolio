// Complete prepared manuscript rendered at authoring time.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { NormalizationRulerFigure, NormalizationMembershipLab, NormalizationGeometryLab, BatchNormalizationStateLab, NormalizationGradientLab, NormalizationMeasuredLab, NormalizationPlacementLab } from '../../components/lesson-labs/NormalizationLabs.jsx';
import NeuralProgram from '../../components/lesson-labs/NeuralProgram.jsx';
export default {
  title: 'Batch, Layer, Group & RMS Normalization',
  readTime: '~65 min read + experiments and practice',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson normalization-lesson">
  <LessonIntro prerequisites="Neural activations, a loss and a gradient. Mean, variance, tensor axes, trainable parameters and stored buffers are explained as they arise." sections={[["a-reference-frame-for-a-collection-of-numbers","A reference frame for a collection of numbers"],["tensor-axes-name-the-collection-before-applying-a-formula","Tensor axes: name the collection before applying a formula"],["layernorm-and-rmsnorm-on-a-feature-vector","LayerNorm and RMSNorm on a feature vector"],["batchnorm-remembers-information-between-forward-passes","BatchNorm remembers information between forward passes"],["the-operation-is-differentiable-including-its-statistics","The operation is differentiable, including its statistics"],["a-complete-cpu-experiment","A complete CPU experiment"],["deeper-placement-information-and-numerical-behavior","Deeper: placement, information, and numerical behavior"],["implement-the-derivative-and-connect-it-to-the-module","Implement the derivative and connect it to the module"],["practice-and-diagnosis","Practice and diagnosis"],["another-explanation-and-the-next-step","Another explanation and the next step"]]}>Name the values that share a statistic before choosing a normalization layer.</LessonIntro>
  <Prose>{""}<strong>{"Explore as you read."}</strong>{" Edit tensor cells, normalization method/group count, offset/scale, epsilon, affine values, mode and running-statistic parameters. Highlight each statistic membership set and show all affected outputs, means/variances and running buffers immediately. Compare changing another example with changing a member of the same group. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose a normalizer and mode by its information dependencies, batch sensitivity and inference state."}</Prose>

<Prose>{"A neural layer may receive numbers whose typical size changes as earlier layers learn. A tanh unit that used to receive values near zero can start receiving values near ten, where its output barely changes and its derivative is small. Normalization layers deliberately rescale collections of intermediate values. Their most important design decision is "}<strong>{"which values share the calculation"}</strong>{"."}</Prose>

<Prose>{"The preceding loss lesson explained how predictions are judged. Here we inspect the intermediate activations that produce those predictions. We will follow a small collection of numbers through normalization, connect the operation to gradients, and compare real training runs."}</Prose>

<Prose>{""}<strong>{"First pass:"}</strong>{" calculate one normalization, explore the axis map, compare BatchNorm's training/evaluation modes, and run the small handwriting experiment. The gradient derivation, residual placement, precision, and distributed sections are deeper branches. You do not need to know convolutions or Transformers to follow the core: the tensor axes are introduced locally."}</Prose>

<H2>{"A reference frame for a collection of numbers"}</H2>

<Prose>{"Suppose a collection contains "}<InlineMath>{"1,3,5,7"}</InlineMath>{". Its mean is four. Subtracting four gives deviations "}<InlineMath>{"-3,-1,1,3"}</InlineMath>{". Their mean squared value is five, so the standard deviation is "}<InlineMath>{"\\sqrt5"}</InlineMath>{". Dividing each deviation by it produces approximately "}<InlineMath>{"-1.342,-.447,.447,1.342"}</InlineMath>{"."}</Prose>

<Prose>{"For a group "}<InlineMath>{"S"}</InlineMath>{" of "}<InlineMath>{"M"}</InlineMath>{" activation values, the centered normalization is"}</Prose>

<div className="neural-equation"><MathBlock>{"\\mu=\\frac1M\\sum_i x_i,\\qquad\nv=\\frac1M\\sum_i(x_i-\\mu)^2,\\qquad\n\\hat x_i=\\frac{x_i-\\mu}{\\sqrt{v+\\epsilon}}."}</MathBlock></div>

<Prose>{"The small positive "}<InlineMath>{"\\epsilon"}</InlineMath>{" prevents a zero denominator. Here variance divides by "}<InlineMath>{"M"}</InlineMath>{", not "}<InlineMath>{"M-1"}</InlineMath>{". It describes the collection being normalized; later we will distinguish the running-variance convention used by PyTorch BatchNorm."}</Prose>

<NormalizationRulerFigure />

<Prose>{"Before any learned affine transformation, the normalized collection has mean zero and variance "}<InlineMath>{"v/(v+\\epsilon)"}</InlineMath>{", which is close to one only when "}<InlineMath>{"v"}</InlineMath>{" is large relative to "}<InlineMath>{"\\epsilon"}</InlineMath>{". The result is not necessarily Gaussian. A bimodal collection remains bimodal."}</Prose>

<Prose>{"Networks often follow the normalization with learned scale and shift:"}</Prose>

<div className="neural-equation"><MathBlock>{"y_i=\\gamma_i\\hat x_i+\\beta_i."}</MathBlock></div>

<Prose>{"These are trainable parameters, not measured means or variances. When the same "}<InlineMath>{"\\gamma,\\beta"}</InlineMath>{" apply across a normalization group, its output mean is "}<InlineMath>{"\\beta"}</InlineMath>{" and variance is "}<InlineMath>{"\\gamma^2v/(v+\\epsilon)"}</InlineMath>{". With different per-feature scales and shifts within the group, use the actual transformed values to calculate those statistics."}</Prose>

<Prose>{"The affine transform lets the network choose useful output scales and offsets. It cannot generally reconstruct the different mean and magnitude removed from every possible input. For example, "}<InlineMath>{"[1,3]"}</InlineMath>{" and "}<InlineMath>{"[11,13]"}</InlineMath>{" produce the same centered normalized vector. A single fixed affine transform sees identical inputs and cannot infer which original offset was present."}</Prose>

<Prose>{"This loss of information is sometimes desirable and sometimes costly. Normalization is a modeling choice, not a universally harmless numerical cleanup."}</Prose>

<H2>{"Tensor axes: name the collection before applying a formula"}</H2>

<Prose>{"A tensor is a multidimensional table. In an image-like representation with shape "}<InlineMath>{"(N,C,H,W)"}</InlineMath>{":"}</Prose>

<ul><li>{""}<InlineMath>{"N"}</InlineMath>{" indexes examples in the current batch."}</li><li>{""}<InlineMath>{"C"}</InlineMath>{" indexes channels: different measurements or learned features at a position."}</li><li>{""}<InlineMath>{"H,W"}</InlineMath>{" index positions in a two-dimensional grid."}</li></ul>

<Prose>{"A channel could initially contain grayscale brightness; inside a model it can contain a learned response. We do not need the convolution operation yet to understand which cells are grouped."}</Prose>

<Prose>{"Consider "}<InlineMath>{"(N,C,H,W)=(2,4,1,2)"}</InlineMath>{". The first example contains channel rows "}<InlineMath>{"[1,2],[3,4],[5,6],[7,8]"}</InlineMath>{"; the second contains "}<InlineMath>{"[9,10],[11,12],[13,14],[15,16]"}</InlineMath>{"."}</Prose>

<NeuralTable caption={"Tensor axes: name the collection before applying a formula"} headers={[<>{"Method and this declared layout"}</>,<>{"One statistics group contains"}</>,<>{"Number of values per group"}</>]} rows={[[<>{"Training BatchNorm2d"}</>,<>{"One channel, all examples and spatial positions"}</>,<>{""}<InlineMath>{"NHW=4"}</InlineMath>{""}</>],[<>{"LayerNorm over "}<InlineMath>{"(C,H,W)"}</InlineMath>{""}</>,<>{"One complete example"}</>,<>{""}<InlineMath>{"CHW=8"}</InlineMath>{""}</>],[<>{"GroupNorm with "}<InlineMath>{"G=2"}</InlineMath>{""}</>,<>{"One example, two adjacent channels and all their positions"}</>,<>{""}<InlineMath>{"(C/G)HW=4"}</InlineMath>{""}</>],[<>{"InstanceNorm2d"}</>,<>{"One example, one channel, all positions"}</>,<>{""}<InlineMath>{"HW=2"}</InlineMath>{""}</>]]} />

<NormalizationMembershipLab />

<Prose>{"Now change only the value nine to nineteen, in the second example. Inspect which methods change the normalized first example. Only training BatchNorm does: its shared reference group crossed the batch axis. In the executed fixture, the largest first-example output change is about .150221 for BatchNorm and exactly zero for the other three methods."}</Prose>

<Prose>{"This is a dependency question, not a claim that BatchNorm is worse. Sharing across examples provides different statistical information and also introduces batch-composition dependence. The "}<a href={"https://arxiv.org/pdf/1803.08494"}>{"Group Normalization paper's formulation and Figure 2"}</a>{" offer another useful view of these grouping choices."}</Prose>

<H3>{"Group limits do not erase parameter differences"}</H3>

<Prose>{"With "}<InlineMath>{"G=1"}</InlineMath>{", GroupNorm uses the same centered statistics as LayerNorm over the complete "}<InlineMath>{"(C,H,W)"}</InlineMath>{" example. With "}<InlineMath>{"G=C"}</InlineMath>{", its statistics match InstanceNorm over "}<InlineMath>{"(H,W)"}</InlineMath>{". The program checks both equalities with common epsilon and identity affine parameters."}</Prose>

<Prose>{"But PyTorch's LayerNorm can learn a distinct scale and shift for every element of its declared normalized shape. GroupNorm learns them per channel and shares them over positions. Thus "}<code>{"GroupNorm(1,C)"}</code>{" and "}<code>{"LayerNorm((C,H,W))"}</code>{" need not represent identical learned functions once their affine parameters differ. Moreover, "}<code>{"LayerNorm(C)"}</code>{" normalizes the "}<strong>{"last dimension"}</strong>{"; it does not automatically find the channel axis in an NCHW tensor. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html"}>{"LayerNorm"}</a>{" and "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.GroupNorm.html"}>{"GroupNorm"}</a>{" document these contracts."}</Prose>

<Prose>{"The number of groups must divide the channel count. Choose a valid count for each layer, then evaluate it. There is no rule that 32 groups is possible or optimal for every channel width."}</Prose>

<H2>{"LayerNorm and RMSNorm on a feature vector"}</H2>

<Prose>{"For a sequence representation "}<InlineMath>{"(B,T,D)"}</InlineMath>{", "}<InlineMath>{"B"}</InlineMath>{" indexes examples, "}<InlineMath>{"T"}</InlineMath>{" token positions, and "}<InlineMath>{"D"}</InlineMath>{" features describing each token. A token can be a word piece, an audio step, or another sequence element. LayerNorm with "}<code>{"normalized_shape=D"}</code>{" calculates statistics across features "}<strong>{"within each token"}</strong>{", independently of other tokens and examples."}</Prose>

<Prose>{"RMSNorm uses the same feature group but a different statistic:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\operatorname{RMSNorm}(x)_i\n=\\gamma_i\\frac{x_i}{\\sqrt{\\frac1D\\sum_jx_j^2+\\epsilon}}."}</MathBlock></div>

<Prose>{"It rescales by the root mean square without subtracting the mean. PyTorch's module has a learned scale and no additive bias. Do not describe every normalization method as centering to zero. "}<a href={"https://arxiv.org/pdf/1910.07467"}>{"The RMSNorm paper, §4"}</a>{" motivates retaining rescaling while removing recentering; its empirical results are not a guarantee for every model."}</Prose>

<Prose>{"With identity affine parameters and "}<InlineMath>{"\\epsilon=10^{-5}"}</InlineMath>{":"}</Prose>

<NeuralTable caption={"LayerNorm and RMSNorm on a feature vector"} headers={[<>{"Input"}</>,<>{"LayerNorm output, approximately"}</>,<>{"RMSNorm output, approximately"}</>]} rows={[[<>{""}<InlineMath>{"[1,3]"}</InlineMath>{""}</>,<>{""}<InlineMath>{"[-1,1]"}</InlineMath>{""}</>,<>{""}<InlineMath>{"[.447213,1.341639]"}</InlineMath>{""}</>],[<>{""}<InlineMath>{"[11,13]"}</InlineMath>{""}</>,<>{""}<InlineMath>{"[-1,1]"}</InlineMath>{""}</>,<>{""}<InlineMath>{"[.913500,1.079591]"}</InlineMath>{""}</>],[<>{""}<InlineMath>{"[-1,1]"}</InlineMath>{""}</>,<>{""}<InlineMath>{"[-1,1]"}</InlineMath>{""}</>,<>{""}<InlineMath>{"[-1,1]"}</InlineMath>{""}</>],[<>{""}<InlineMath>{"[5,5]"}</InlineMath>{""}</>,<>{""}<InlineMath>{"[0,0]"}</InlineMath>{""}</>,<>{""}<InlineMath>{"[1,1]"}</InlineMath>{""}</>],[<>{""}<InlineMath>{"[0,0]"}</InlineMath>{""}</>,<>{""}<InlineMath>{"[0,0]"}</InlineMath>{""}</>,<>{""}<InlineMath>{"[0,0]"}</InlineMath>{""}</>]]} />

<Prose>{"These values were calculated by the accompanying program. RMSNorm retains information about the vector's common offset relative to its magnitude; LayerNorm removes a common offset. Both reduce sensitivity to positive overall scaling, exactly when epsilon is zero and denominators are nonzero, approximately when epsilon is small compared with the relevant squared scale. Negative scaling reverses signs rather than leaving outputs unchanged."}</Prose>

<NormalizationGeometryLab />

<Prose>{"For a causal sequence model, an output at position "}<InlineMath>{"t"}</InlineMath>{" must not depend on future inputs. Per-token normalization over "}<InlineMath>{"D"}</InlineMath>{" respects that boundary. Normalizing over "}<InlineMath>{"(T,D)"}</InlineMath>{" mixes token positions and can introduce a future dependency even if attention itself has a causal mask. Padding included in a reduction can likewise change its statistics. Check the axes rather than assuming the layer's name guarantees the desired dependencies."}</Prose>

<H2>{"BatchNorm remembers information between forward passes"}</H2>

<Prose>{"Training BatchNorm computes current-batch statistics. With running statistics enabled, it also updates stored estimates that evaluation will use. That makes it a "}<strong>{"stateful"}</strong>{" layer: the same parameters can behave differently depending on stored buffers and mode."}</Prose>

<Prose>{"In PyTorch's ordinary fixed-momentum convention,"}</Prose>

<div className="neural-equation"><MathBlock>{"\\mu_{\\mathrm{run,new}}=(1-m)\\mu_{\\mathrm{run,old}}+m\\mu_{\\mathrm{batch}}."}</MathBlock></div>

<Prose>{"Here "}<InlineMath>{"m"}</InlineMath>{" is the weight on the new statistic. For the running variance, PyTorch uses the batch's variance with denominator "}<InlineMath>{"M-1"}</InlineMath>{", while the training normalization itself uses denominator "}<InlineMath>{"M"}</InlineMath>{". This is a library contract; an exponential average is not a guarantee of the exact whole-dataset variance. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html"}>{"BatchNorm2d's documentation"}</a>{" specifies the behavior and the "}<code>{"track_running_stats=False"}</code>{" exception."}</Prose>

<Prose>{"Start with running mean zero and running variance one. A channel contains "}<InlineMath>{"[1,3,5,7]"}</InlineMath>{", mean four and within-batch variance five. The corrected variance is "}<InlineMath>{"5\\times4/3=20/3"}</InlineMath>{". With momentum .1:"}</Prose>

<NeuralTable caption={"BatchNorm remembers information between forward passes"} headers={[<>{"Item"}</>,<>{"After one training forward pass"}</>]} rows={[[<>{"Running mean"}</>,<>{""}<InlineMath>{".9(0)+.1(4)=.4"}</InlineMath>{""}</>],[<>{"Running variance"}</>,<>{""}<InlineMath>{".9(1)+.1(20/3)=1.566667"}</InlineMath>{""}</>],[<>{"Training-normalized first value"}</>,<>{""}<InlineMath>{"(1-4)/\\sqrt{5+10^{-5}}\\approx-1.341639"}</InlineMath>{""}</>],[<>{"Same value in evaluation mode"}</>,<>{""}<InlineMath>{"(1-.4)/\\sqrt{1.566667+10^{-5}}\\approx .479360"}</InlineMath>{""}</>]]} />

<Prose>{"Calling "}<code>{"eval()"}</code>{" does not force these two functions to agree. It switches to the stored estimates, which are still immature after one pass. A mode bug, insufficiently representative running statistics, a distribution change, and incorrect preprocessing are different diagnoses."}</Prose>

<BatchNormalizationStateLab />

<H3>{"One image is not always one value"}</H3>

<Prose>{"Training BatchNorm2d with shape "}<InlineMath>{"(1,C,H,W)"}</InlineMath>{" has "}<InlineMath>{"HW"}</InlineMath>{" values per channel. For a single channel with spatial values "}<InlineMath>{"[1,3]"}</InlineMath>{", it produces approximately "}<InlineMath>{"[-1,1]"}</InlineMath>{"; the variance is not zero. With exactly one value per channel, PyTorch's training BatchNorm rejects the input rather than silently treating it as a normal training case."}</Prose>

<Prose>{"Spatial values can be correlated. Having a thousand adjacent pixels is not equivalent to having a thousand independent images. This explains why the useful amount of batch information depends on the task and representation, not only an advertised batch-size threshold."}</Prose>

<Prose>{"Three switches control three different things:"}</Prose>

<NeuralTable caption={"One image is not always one value"} headers={[<>{"Action"}</>,<>{"What it changes"}</>,<>{"What it does not do"}</>]} rows={[[<>{""}<code>{"model.eval()"}</code>{""}</>,<>{"Mode-sensitive layer behavior; ordinary BatchNorm uses running statistics"}</>,<>{"Disable gradient recording"}</>],[<>{""}<code>{"torch.no_grad()"}</code>{""}</>,<>{"Autograd recording inside the context"}</>,<>{"Stop training-mode BatchNorm buffer updates"}</>],[<>{""}<code>{"parameter.requires_grad_(False)"}</code>{""}</>,<>{"Gradient calculation for that parameter"}</>,<>{"Freeze all module buffers or switch modes"}</>]]} />

<Prose>{"A forward pass alone does not perform an optimizer update. If a model adapts BatchNorm statistics to a deployment sample, that still consumes information and changes the predictor even without changing weights. Use an authorized adaptation set and evaluate the resulting protocol honestly. Do not quietly recalibrate on a final test and call it untouched."}</Prose>

<Prose>{"Ordinary gradient accumulation over eight microbatches does not give BatchNorm one eight-times-larger batch: each forward pass uses its own statistics and updates buffers. SyncBatchNorm can combine statistics across participating devices for a simultaneous step, at a communication cost. It does not automatically pool sequential accumulation steps. Whether to retain BatchNorm, synchronize it, freeze its state, or change the architecture must be evaluated with the actual pretrained model and training protocol."}</Prose>

<H2>{"The operation is differentiable, including its statistics"}</H2>

<Prose>{"The mean and variance depend on every value in their group. Detaching them from autograd changes the gradient. For one centered group, write "}<InlineMath>{"s=\\sqrt{v+\\epsilon}"}</InlineMath>{", upstream derivatives "}<InlineMath>{"d_i=\\partial L/\\partial y_i"}</InlineMath>{", and "}<InlineMath>{"u_i=d_i\\gamma_i"}</InlineMath>{". Then"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{\\partial L}{\\partial x_i}\n=\\frac1s\\left[u_i-\\operatorname{mean}(u)\n-\\hat x_i\\operatorname{mean}(u\\hat x)\\right]."}</MathBlock></div>

<Prose>{"The two subtraction terms account for changing the group's center and scale. The formula is valid with the stated variance and epsilon convention; do not replace "}<InlineMath>{"\\hat x"}</InlineMath>{" with an independently sampled normalized vector."}</Prose>

<Prose>{"For RMSNorm, with "}<InlineMath>{"r=\\sqrt{\\operatorname{mean}(x^2)+\\epsilon}"}</InlineMath>{", there is no centering correction:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{\\partial L}{\\partial x_i}\n=\\frac{u_i}{r}-\\frac{x_i\\,\\operatorname{mean}(ux)}{r^3}."}</MathBlock></div>

<Prose>{"Scale gradients accumulate "}<InlineMath>{"d_i\\hat x_i"}</InlineMath>{" across uses of each shared parameter, and shift gradients accumulate "}<InlineMath>{"d_i"}</InlineMath>{". These reductions follow the affine parameter's sharing pattern, which can differ from the statistics group's axes."}</Prose>

<Prose>{"Take "}<InlineMath>{"x=[1,3]"}</InlineMath>{", LayerNorm, "}<InlineMath>{"\\gamma=[1,2]"}</InlineMath>{", "}<InlineMath>{"\\beta=[0,0]"}</InlineMath>{", target "}<InlineMath>{"[0,1]"}</InlineMath>{", and mean squared error. The initial output is about "}<InlineMath>{"[-1,2]"}</InlineMath>{", loss .999985. The scale gradients are approximately "}<InlineMath>{"[.999990,.999985]"}</InlineMath>{" and shift gradients "}<InlineMath>{"[-.999995,.999990]"}</InlineMath>{". One gradient step of .1 updating only scale and shift produces output approximately "}<InlineMath>{"[-.799997,1.799993]"}</InlineMath>{" and loss .639992. Those are executed values, with "}<InlineMath>{"x"}</InlineMath>{" held fixed for the update."}</Prose>

<Prose>{"The tiny input gradient in this two-feature example is not evidence that the engine is broken. Away from ties, centering and dividing by the spread of two values leaves almost only their order when epsilon is small. The affine parameters can still learn. Use more features and changed inputs when investigating broader gradients."}</Prose>

<NormalizationGradientLab />

<H2>{"A complete CPU experiment"}</H2>

<Prose>{"Download "}<a href={"/learn-assets/batch-layer-group-rms-normalization/normalization-experiments.py"}>{"normalization-experiments.py"}</a>{" and "}<a href={"/learn-assets/batch-layer-group-rms-normalization/digits-400.csv"}>{"digits-400.csv"}</a>{" into the same directory. The program contains readable centered, group, RMS, and stateful BatchNorm formulas, comparison fixtures, and twelve complete training runs. It uses no pretrained model or dataset download during training."}</Prose>

<NeuralProgram topic="normalization" /><p><a href="/learn-assets/batch-layer-group-rms-normalization/recorded-output.txt">Recorded complete-program output</a> · <a href="/learn-assets/batch-layer-group-rms-normalization/snippet-output.txt">Precision snippet output</a></p>

<CodeBlock language={"text"}>{"python -m venv .venv\n.venv\\Scripts\\python -m pip install numpy==2.3.5 torch==2.14.0 scikit-learn==1.9.1\n.venv\\Scripts\\python normalization-experiments.py"}</CodeBlock>

<Prose>{"Use "}<code>{".venv/bin/python"}</code>{" on macOS/Linux. The recorded run used Python 3.12.14, torch 2.14.0+cpu, and one CPU thread. Implementation replay in the declared existing environment reproduced all saved results exactly; no fresh package installation was needed."}</Prose>

<Prose>{"The real data has 400 UCI handwritten digit specimens, 40 per class, each with 64 pixel values from zero to 16. We divide by 16 and use the same stratified 280/120 development split as the earlier neural lessons. The "}<a href={"/learn-assets/batch-layer-group-rms-normalization/data-provenance.md"}>{"provenance record"}</a>{" explains why this is not an official UCI or writer-independent benchmark."}</Prose>

<Prose>{"The model is "}<code>{"64 → Linear(32) → normalization → tanh → Linear(10)"}</code>{". The normalized vector contains learned features, not spatial convolution channels. Compare no normalization, BatchNorm1d, LayerNorm, and RMSNorm; all affine defaults and epsilon are explicitly set or documented in code. Train with ordinary SGD at .1 for 50 epochs, ten batches of 28 per epoch. The example order and initial linear weights are matched for each seed. Use seeds one, two, and three."}</Prose>

<Prose>{"The program switches to evaluation mode for reported training and validation loss. This matters for BatchNorm: the reported training-set loss uses stored running statistics and is not the sequence of instantaneous batch losses optimized during that epoch. The code records epochs0,1,5,20,50 rather than fabricating smooth curves between arbitrary values."}</Prose>

<NeuralTable caption={"A complete CPU experiment"} headers={[<>{"Seed"}</>,<>{"Normalization"}</>,<>{"Training-set CE in evaluation mode"}</>,<>{"Validation CE"}</>,<>{"Correct /120"}</>]} rows={[[<>{"1"}</>,<>{"None"}</>,<>{".101865"}</>,<>{".162959"}</>,<>{"116"}</>],[<>{"1"}</>,<>{"Batch"}</>,<>{".021480"}</>,<>{".116648"}</>,<>{"117"}</>],[<>{"1"}</>,<>{"Layer"}</>,<>{".023192"}</>,<>{".118822"}</>,<>{"116"}</>],[<>{"1"}</>,<>{"RMS"}</>,<>{".023227"}</>,<>{".116965"}</>,<>{"116"}</>],[<>{"2"}</>,<>{"None"}</>,<>{".095411"}</>,<>{".163503"}</>,<>{"116"}</>],[<>{"2"}</>,<>{"Batch"}</>,<>{".019784"}</>,<>{".128102"}</>,<>{"117"}</>],[<>{"2"}</>,<>{"Layer"}</>,<>{".022219"}</>,<>{".121958"}</>,<>{"116"}</>],[<>{"2"}</>,<>{"RMS"}</>,<>{".021885"}</>,<>{".117832"}</>,<>{"116"}</>],[<>{"3"}</>,<>{"None"}</>,<>{".096612"}</>,<>{".174415"}</>,<>{"116"}</>],[<>{"3"}</>,<>{"Batch"}</>,<>{".019923"}</>,<>{".142852"}</>,<>{"117"}</>],[<>{"3"}</>,<>{"Layer"}</>,<>{".021124"}</>,<>{".146711"}</>,<>{"115"}</>],[<>{"3"}</>,<>{"RMS"}</>,<>{".021135"}</>,<>{".145417"}</>,<>{"115"}</>]]} />

<Prose>{"These are actual outputs for a fixed small protocol. All four variants learned; the unnormalized model did not diverge. Normalization reduced the final losses in these runs, while the count differences were small and seed dependent. This experiment does not prove a universal ranking, a speedup, or a requirement for normalization. Learning rates and architectures were not separately tuned to each variant, and no untouched final test was used to endorse a selected model."}</Prose>

<Prose>{"The formula fixtures also compare outputs against torch modules using float64. Maximum forward discrepancies were below "}<InlineMath>{"3.2\\times10^{-15}"}</InlineMath>{". The stateful BatchNorm function matched running means and variances exactly on its fixture and evaluation output within "}<InlineMath>{"1.8\\times10^{-15}"}</InlineMath>{". Layer and group input-gradient comparisons were within "}<InlineMath>{"1.4\\times10^{-15}"}</InlineMath>{". These are focused author checks, not certification of arbitrary production inputs."}</Prose>

<NormalizationMeasuredLab />

<H2>{"Deeper: placement, information, and numerical behavior"}</H2>

<H3>{"A residual connection is an extra path"}</H3>

<Prose>{"A residual block adds a learned change "}<InlineMath>{"F(x)"}</InlineMath>{" to its input: "}<InlineMath>{"x+F(x)"}</InlineMath>{". Detailed residual architectures come later. This small definition is enough to compare normalization placement:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\text{post-norm: }y=\\operatorname{LN}(x+F(x)),\\qquad\n\\text{pre-norm: }y=x+F(\\operatorname{LN}(x))."}</MathBlock></div>

<Prose>{"If "}<InlineMath>{"F=0"}</InlineMath>{", pre-norm returns "}<InlineMath>{"x"}</InlineMath>{"; post-norm returns "}<InlineMath>{"\\operatorname{LN}(x)"}</InlineMath>{". The positions are not interchangeable wrappers. In pre-norm, one direct derivative path is an identity. The total derivative also includes the learned branch and can still amplify or cancel directions."}</Prose>

<Prose>{""}<a href={"https://proceedings.mlr.press/v119/xiong20b.html"}>{"Xiong et al."}</a>{" analyze how placement affects initialization-time gradients and warm-up in specified Transformer settings. That supports a useful design explanation, not a guarantee that arbitrary pre-norm networks need no warm-up or that post-norm fails above a fixed depth. Swapping placement in a trained checkpoint changes its function."}</Prose>

<NormalizationPlacementLab />

<H3>{"What normalization explanations can claim"}</H3>

<Prose>{"The original BatchNorm paper framed its motivation using changing internal input distributions. Later work demonstrated that this explanation alone does not account for its benefits and studied smoother optimization behavior. "}<a href={"https://proceedings.mlr.press/v37/ioffe15.html"}>{"Ioffe and Szegedy"}</a>{" and "}<a href={"https://arxiv.org/abs/1805.11604"}>{"Santurkar et al."}</a>{" are useful to read together. A careful explanation separates an operation we can calculate, a mechanism supported under stated analysis, and empirical results that depend on architecture and training."}</Prose>

<Prose>{"Normalization can alter parameter scale sensitivity, gradients, and the stochasticity introduced by batch composition. It need not make the full loss landscape globally smooth in every model, eliminate the need for good initialization, or guarantee a larger learning rate is safe."}</Prose>

<H3>{"Precision and epsilon"}</H3>

<Prose>{"Epsilon is added to a variance or mean square, so it shares those squared units. Smaller epsilon can preserve more scale sensitivity at tiny magnitudes and can also produce larger inverse denominators. It is not automatically safer."}</Prose>

<Prose>{"Floating-point "}<strong>{"range"}</strong>{" and "}<strong>{"precision"}</strong>{" are different. BF16 can represent small numbers such as "}<InlineMath>{"10^{-5}"}</InlineMath>{"; its coarse spacing near one does not imply that every smaller number is zero. Squaring a large FP16 activation can overflow before epsilon has any chance to help. For a hand-written low-precision RMS implementation, promote values before squaring and reducing:"}</Prose>

<CodeBlock language={"python"}>{"import torch\n\nx = torch.tensor([[300., 400.]], dtype=torch.float16)\nscale = torch.ones(2, dtype=torch.float32)\nworking = x.float()\ny = (working * torch.rsqrt(working.square().mean(-1, keepdim=True) + 1e-5) * scale)\ny = y.to(x.dtype)\nprint(y)  # derived: approximately [.8486, 1.1318] after float16 rounding"}</CodeBlock>

<Prose>{"This snippet states its output dtype; it is not an exact emulation of every autocast or fused-kernel rule. The packet's mathematical checks use float64 and training uses float32. Consult the actual library/version and keep checkpoint epsilon consistent. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.RMSNorm.html"}>{"PyTorch RMSNorm"}</a>{" documents the current default when epsilon is unspecified; our experiment supplies "}<InlineMath>{"10^{-5}"}</InlineMath>{" explicitly."}</Prose>

<H3>{"Costs and related ideas"}</H3>

<Prose>{"An activation normalization examines order one value per element, with reductions and affine work. Linear complexity does not mean free: reductions, memory traffic, synchronization, and kernel launches can matter. For "}<InlineMath>{"(B,T,D)=(8,8192,8192)"}</InlineMath>{", there are 536,870,912 elements. One float32 read is 2 GiB; one two-byte read is 1 GiB. These are traffic calculations, not a measured end-to-end speedup."}</Prose>

<Prose>{"Fused implementations can reduce intermediate allocations and memory traversals. The exact passes and runtime depend on kernel, shape, dtype, hardware and compiler. RMSNorm's simpler statistic can save work, but fewer source-code operations do not establish a universal percentage improvement."}</Prose>

<Prose>{"InstanceNorm's per-image, per-channel grouping has a useful connection to stylization: changing global contrast or channel offsets need not change normalized patterns. That property helps some image transformations but can erase intensity information needed elsewhere. "}<a href={"https://arxiv.org/abs/1607.08022"}>{"Ulyanov et al."}</a>{" introduced the method in a fast-stylization setting. Weight normalization is a different family: it writes a weight vector as "}<InlineMath>{"w=g\\,v/\\|v\\|"}</InlineMath>{", separating magnitude and direction rather than computing activation statistics. "}<a href={"https://arxiv.org/abs/1602.07868"}>{"Salimans and Kingma"}</a>{" is an alternate deeper route."}</Prose>

<H2>{"Implement the derivative and connect it to the module"}</H2>

<Prose>{"The "}<a href={"/learn-assets/batch-layer-group-rms-normalization/normalization-experiments.py"}>{"complete experiment above"}</a>{" already implements centered normalization, RMS normalization, grouped reshaping and BatchNorm's train/eval running-state update from tensor primitives. Read those functions before the model-fitting code. They own the forward mechanism: no normalization module is called inside them. The reference modules in the fixture section check the result, including BatchNorm's different training and stored variance denominators. We do not need to recopy that forward implementation into another neural architecture lesson."}</Prose>

<Prose>{"The remaining piece is an explicit backward operation. Let "}<InlineMath>{"h"}</InlineMath>{" be the normalized values, "}<InlineMath>{"r"}</InlineMath>{" the inverse scale, and "}<InlineMath>{"g"}</InlineMath>{" the gradient arriving "}<strong>{"after multiplying by the affine scale"}</strong>{". For a centered collection, the input gradient is"}</Prose>

<div className="neural-equation"><MathBlock>{"dx=r\\bigl(g-\\operatorname{mean}(g)-h\\operatorname{mean}(gh)\\bigr)."}</MathBlock></div>

<Prose>{"The direct path contributes "}<InlineMath>{"g"}</InlineMath>{". Changing the shared mean contributes the subtraction of its mean. Changing the shared variance contributes the last correction. Each mean is over exactly the same collection as the forward operation. For RMSNorm there is no centering path, so omit "}<InlineMath>{"\\operatorname{mean}(g)"}</InlineMath>{", retaining the scale correction. The formula includes epsilon through "}<InlineMath>{"h"}</InlineMath>{" and "}<InlineMath>{"r"}</InlineMath>{"; it does not assume that epsilon is zero."}</Prose>

<CodeBlock language={"python"}>{"import numpy as np\n\ndef normalize(values, axes, epsilon=1e-5, centered=True):\n    mean = values.mean(axis=axes, keepdims=True) if centered else 0.\n    deviation = values - mean\n    inverse_scale = 1 / np.sqrt((deviation ** 2).mean(axis=axes, keepdims=True) + epsilon)\n    normalized = deviation * inverse_scale\n    return normalized, (normalized, inverse_scale, axes, centered)\n\ndef backward(upstream, cache):\n    normalized, inverse_scale, axes, centered = cache\n    correlated = (upstream * normalized).mean(axis=axes, keepdims=True)\n    gradient = upstream - normalized * correlated\n    if centered:\n        gradient = gradient - upstream.mean(axis=axes, keepdims=True)\n    return inverse_scale * gradient\n\nx = np.array([[1., 3., -2.]])\nh, cache = normalize(x, (1,))\ndx = backward(np.array([[.3, -.2, .8]]), cache)\nprint(np.round(h, 6))\nprint(np.round(dx.sum(axis=1), 12))  # [0.]"}</CodeBlock>

<Prose>{"These are vector-Jacobian products: the routine computes the effect of one incoming gradient without constructing a dense Jacobian. For "}<InlineMath>{"M"}</InlineMath>{" input values it uses "}<InlineMath>{"O(M)"}</InlineMath>{" arithmetic and stored normalized activations; a dense "}<InlineMath>{"M\\times M"}</InlineMath>{" derivative matrix would waste space. Broadcasting keeps the group means small. NumPy's reductions and elementwise operations are the stated primitive boundary; this is not a custom GPU kernel or a mixed-precision emulation."}</Prose>

<Prose>{"For "}<InlineMath>{"y=\\gamma h+\\beta"}</InlineMath>{", send "}<code>{"upstream * gamma"}</code>{" into the normalization VJP. The scale gradient sums "}<code>{"upstream * h"}</code>{" over the positions that share each scale parameter, and the shift gradient sums "}<code>{"upstream"}</code>{" over those same positions. Those parameter-sharing axes need not equal the normalization axes. A token LayerNorm with one scale per feature sums parameter gradients over tokens, while its normalization statistics reduce features. Conflating these reductions gives a plausible-looking tensor with the wrong derivative."}</Prose>

<Prose>{"Download "}<a href={"/learn-assets/batch-layer-group-rms-normalization/normalization-backward.py"}>{"normalization-backward.py"}</a>{", use the earlier environment, and run "}<code>{"python normalization-backward.py"}</code>{". It supplies all fixtures, matches forward values and manual input gradients against BatchNorm, LayerNorm, GroupNorm, InstanceNorm and RMSNorm, and compares the LayerNorm affine gradients. It also checks the manual derivative by finite differences, independently of PyTorch's backward engine. "}<a href={"/learn-assets/batch-layer-group-rms-normalization/normalization-backward-output.json"}>{"Recorded output"}</a>{" gives the actual errors. The existing forward experiment remains the owner of running mean/variance and evaluation-mode checks."}</Prose>

<NeuralProgram topic="normalization-backward" title="Read the manual normalization derivatives and library checks" />

<NeuralTable caption={"Implement the derivative and connect it to the module"} headers={[<>{"Choice in the scratch route"}</>,<>{"Corresponding module decision"}</>]} rows={[[<>{"NCHW axes "}<code>{"(0, 2, 3)"}</code>{""}</>,<>{"BatchNorm2d training statistics; evaluation uses stored statistics instead"}</>],[<>{"NCHW axes "}<code>{"(1, 2, 3)"}</code>{""}</>,<>{"LayerNorm with normalized shape "}<code>{"(C, H, W)"}</code>{""}</>],[<>{"Reshape channels into groups; reduce channels-per-group and spatial axes"}</>,<>{"GroupNorm with a divisor of channel count"}</>],[<>{"NCHW axes "}<code>{"(2, 3)"}</code>{""}</>,<>{"InstanceNorm2d without running-stat evaluation"}</>],[<>{"Last feature axis, no mean subtraction"}</>,<>{"RMSNorm with matching normalized shape and explicit epsilon"}</>]]} />

<Prose>{"The VJP above describes "}<strong>{"statistics computed from the current input"}</strong>{". BatchNorm in evaluation mode has fixed stored statistics: its input gradient is upstream times affine scale divided by "}<InlineMath>{"\\sqrt{\\text{running variance}+\\epsilon}"}</InlineMath>{". Do not apply the training VJP to evaluation. The variance-buffer update itself is not a differentiable parameter update. The "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html"}>{"BatchNorm contract"}</a>{" and "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.RMSNorm.html"}>{"RMSNorm contract"}</a>{" explain the relevant library choices; the programs set epsilon explicitly and compare in float64."}</Prose>

<Prose>{""}<strong>{"Extension — fit an affine scale manually."}</strong>{" For the two-token fixture in the downloaded program, take one small gradient step on gamma and beta using its explicit gradients, hold the input fixed and compare the new output with a LayerNorm module whose parameters received the same update. Then switch to RMSNorm and remove the centering correction. Compare both the input derivative and its sum; the LayerNorm zero-sum property should not be assumed for RMSNorm."}</Prose>

<details><summary>Hint</summary>

<Prose>{"The loss used for the gradient check is the sum of output times the fixed incoming coefficients. Its derivative with respect to output is those coefficients, not an additional mean-reduction factor."}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{""}<strong>{"Solution:"}</strong>{" use "}<code>{"gamma_next = gamma - learning_rate * (incoming * h).sum(0)"}</code>{" and "}<code>{"beta_next = beta - learning_rate * incoming.sum(0)"}</code>{". The new output is "}<code>{"gamma_next * h + beta_next"}</code>{". Copy these into the module before comparing. In the RMS variant, recompute "}<code>{"h"}</code>{" without centering and call the uncentered backward rule; do not reuse the old centered cache. Rebuilding a whole autodiff engine would obscure this new dependency, so reuse "}<a href={"/learn/path/full-curriculum/backpropagation-automatic-differentiation?module=deep-learning-fundamentals"}>{"the preceding Backpropagation lesson"}</a>{" for graph composition."}</Prose>

</details>

<H2>{"Practice and diagnosis"}</H2>

<section className="neural-practice"><Prose>{""}<strong>{"Different grouping."}</strong>{" A tensor has shape "}<InlineMath>{"(2,6,3,4)"}</InlineMath>{". How many values contribute to one BatchNorm, GroupNorm with three groups, and InstanceNorm statistic?"}</Prose><details><summary>Hint</summary><Prose>{"hold the unreduced indices fixed."}</Prose></details><details><summary>Worked solution</summary><Prose>{"BatchNorm24; GroupNorm24 per example/group; InstanceNorm12. Equal group sizes do not mean equal memberships."}</Prose></details></section>

<section className="neural-practice"><Prose>{""}<strong>{"Fix an axis bug."}</strong>{" With shape "}<InlineMath>{"(2,5,8)"}</InlineMath>{", a causal model uses LayerNorm((5,8)). An early output changes when only the last token changes. Explain and repair the dependency."}</Prose><details><summary>Worked solution</summary><Prose>{"the normalization includes time. LayerNorm(8) normalizes each token's features separately; also inspect other layers for future dependencies. The change affects the model, so a trained checkpoint requires an appropriate retraining/evaluation plan."}</Prose></details></section>

<section className="neural-practice"><Prose>{""}<strong>{"Different running state."}</strong>{" Start running mean2 and variance4. A batch has mean6, population variance9, and four values. Use momentum .25."}</Prose><details><summary>Worked solution</summary><Prose>{"new mean3; corrected batch variance12, so running variance6. Its training denominator still uses9+epsilon. In evaluation, an input6 uses(6−3)/sqrt(6+epsilon)."}</Prose></details></section>

<section className="neural-practice"><Prose>{""}<strong>{"Predict a null."}</strong>{" Add the same constant to all values in one centered normalization group with identity affine. What changes?"}</Prose><details><summary>Worked solution</summary><Prose>{"the mean shifts by that constant; deviations, variance and outputs remain unchanged. Changing just one member generally changes other members' normalized values. RMSNorm is not invariant to the same offset."}</Prose></details></section>

<section className="neural-practice"><Prose>{""}<strong>{"Follow two kinds of state."}</strong>{" Under "}<code>{"torch.no_grad()"}</code>{", a model in training mode changes its BatchNorm running mean. Is this evidence that the optimizer ran?"}</Prose><details><summary>Worked solution</summary><Prose>{"no; the buffer update is part of the forward operation. Inspect parameters and buffers separately. "}<code>{"eval()"}</code>{" changes ordinary BatchNorm's statistics source; gradient recording is a different switch."}</Prose></details></section>

<section className="neural-practice"><Prose>{""}<strong>{"Change the real experiment."}</strong>{" Reduce batch size from28 to14, keeping50epochs. Predict the number of optimizer steps, running-stat updates, and why this is not a comparison that changes only the number of values in a mean."}</Prose><details><summary>Worked solution</summary><Prose>{"both steps and BatchNorm updates double from500 to1000. A controlled study must declare whether it matches epochs, updates, data exposure, or compute and may need more than one comparison."}</Prose></details></section>

<section className="neural-practice"><Prose>{""}<strong>{"Check apparent equivalence."}</strong>{" GroupNorm(1,4) and LayerNorm((4,1,2)) agree with identity affine parameters. Give a LayerNorm affine setting that GroupNorm's per-channel affine cannot match on arbitrary inputs with fixed shared statistics."}</Prose><details><summary>Worked solution</summary><Prose>{"give the two positions of one channel different shifts or scales. GroupNorm shares that channel's affine parameters across positions."}</Prose></details></section>

<section className="neural-practice"><Prose>{""}<strong>{"Choose a diagnosis from evidence."}</strong>{" Validation becomes worse after changing cameras. Someone proposes running all test images through training-mode BatchNorm. Explain what information this uses and what should happen first."}</Prose><details><summary>Worked solution</summary><Prose>{"it adapts the predictor to the test input distribution and changes buffers; even without labels it is a changed evaluation protocol. First check preprocessing, mode, source/target distributions and representative development data; any adaptation should use a declared allowed dataset and be assessed under the intended deployment protocol."}</Prose></details></section>

<H2>{"Another explanation and the next step"}</H2>

<Prose>{"Use "}<a href={"https://d2l.ai/chapter_convolutional-modern/batch-norm.html"}>{"Dive into Deep Learning's Batch Normalization chapter"}</a>{" for an alternate derivation and model example. Its from-scratch teaching code uses gradient-recording state as a shortcut for mode; keep the explicit train/eval versus no_grad distinction from this lesson when using real torch modules. The primary GroupNorm figure is especially useful for the axis investigation; the LayerNorm and RMSNorm papers explain why removing batch dependence and removing centering are separate decisions. These resources supplement the local lesson rather than supplying missing prerequisites."}</Prose>

<Prose>{"Next is "}<strong>{"Transfer Learning and Fine-Tuning Strategies"}</strong>{". Reusing a learned network means deciding which weights and states are allowed to change. BatchNorm's distinction between trainable affine parameters, running buffers, and mode will be particularly useful there."}</Prose>
  <p><a href="/learn/path/full-curriculum/transfer-learning-fine-tuning-strategies?module=deep-learning-fundamentals">Continue to Transfer Learning and Fine-Tuning Strategies</a></p>
  </div>
};
