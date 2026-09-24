// Full prepared revision-3 manuscript rendered statically; no Markdown parser in the browser.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { ConvolutionPatchLab, ConvolutionChannelsLab, ConvolutionUpdateLab, ConvolutionGeometryLab, ConvolutionPoolingLab, ConvolutionReceptiveLab, ConvolutionMeasuredLab, ConvolutionInfluenceLab, ConvolutionTransposeLab, ConvolutionShiftLab, ConvolutionPatchMatrixFigure, ConvolutionProgram } from '../../components/lesson-labs/ConvolutionLabs.jsx';
export default {
  title: 'Convolution, Pooling & Receptive Fields',
  readTime: '~65 min read + experiments and practice; optional implementation branches ~30 min',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson convolution-lesson">
    <LessonIntro prerequisites="Multiplication, sums and array indexing. Weighted sums, gradients and the objective are refreshed locally; earlier neural and tensor lessons provide deeper prerequisites." sections={[["1-a-small-rule-that-travels","1. A small rule that travels"],["2-channels-are-several-measurements-at-each-location","2. Channels are several measurements at each location"],["3-how-a-shared-filter-learns","3. How a shared filter learns"],["4-plan-the-geometry-before-stacking-layers","4. Plan the geometry before stacking layers"],["5-pooling-summarize-then-notice-what-disappeared","5. Pooling: summarize, then notice what disappeared"],["6-receptive-fields-where-can-this-number-get-information","6. Receptive fields: where can this number get information?"],["7-build-and-inspect-a-real-digit-classifier","7. Build and inspect a real digit classifier"],["8-optional-reach-influence-and-shifts","8. Optional: reach, influence and shifts"],["9-optional-the-reverse-operation-is-a-scatter-not-an-undo","9. Optional: the reverse operation is a scatter, not an undo"],["10-optional-implement-the-same-math-efficiently","10. Optional: implement the same math efficiently"],["implement-the-pullback-and-batch-the-arithmetic","Implement the pullback and batch the arithmetic"],["11-practice-construct-diagnose-transfer","11. Practice: construct, diagnose, transfer"],["references-and-other-ways-to-learn","References and other ways to learn"]]}>Follow one patch, one shared update and a complete digit classifier before the deeper influence, transpose and implementation branches.</LessonIntro>
<Prose>{""}<strong>{"Explore as you read."}</strong>{" Edit image/kernel cells, stride/dilation/padding, pooling inputs, shared-weight targets/rate and receptive-field threshold. Synchronize the selected patch, products, output map, transpose contributions, gradient accumulation and ancestry paths. Geometry edits visibly change output size, alignment and holes rather than only a summary label. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose window geometry and pooling from reach, alignment, information loss and update behavior."}</Prose>

<Prose>{"A handwritten 7 can move a little to the right and still be a 7. Its strokes are local, but their arrangement matters: a short horizontal stroke and a long diagonal belong together. How can a neural network use those facts instead of learning an unrelated detector for every possible pixel position?"}</Prose>

<Prose>{"Convolution reuses a small calculation across a grid. Pooling summarizes nearby values. A receptive field tells us where one result can obtain information. Together, these ideas let us reason about an image model as a sequence of visible operations."}</Prose>

<Prose>{""}<strong>{"First pass:"}</strong>{" follow one patch through a filter, learn how that filter changes, plan the sizes, and train the small digit classifier in sections 1–7. You should then be able to build and diagnose a basic CNN. Sections 8–10 are optional deeper routes into influence, reconstruction and efficient execution. The practice separates core readiness from those extensions."}</Prose>

<Prose>{"You need multiplication, sums and array indexing. We refresh neural terminology as it appears: a parameter is a learned number, an activation is a computed value, and a gradient says how a small change affects a chosen result. Our task is to predict one of ten digit labels from an 8×8 intensity image. A dense network is the baseline; convolution is a different useful constraint, not a promise to beat it."}</Prose>

<H2>{"1. A small rule that travels"}</H2>

<Prose>{"Consider this image and a 2×2 filter:"}</Prose>

<NeuralTable caption={"1. A small rule that travels"} headers={[<>{"Image"}</>,<>{"Filter"}</>]} rows={[[<>{"1, 2, 0; 0, 1, 3; 2, 1, 0"}</>,<>{"1, −1; 0, 1"}</>]]} />

<Prose>{"Put the filter over the top-left patch. Multiply corresponding cells and add:"}</Prose>

<div className="neural-equation"><MathBlock>{"1(1)+2(-1)+0(0)+1(1)=0."}</MathBlock></div>

<Prose>{"Move it one column right, keeping the "}<strong>{"same four weights"}</strong>{":"}</Prose>

<div className="neural-equation"><MathBlock>{"2(1)+0(-1)+1(0)+3(1)=5."}</MathBlock></div>

<Prose>{"Repeating for the second row produces "}<InlineMath>{"\\begin{bmatrix}0&5\\\\0&-2\\end{bmatrix}"}</InlineMath>{". This output grid is a "}<strong>{"feature map"}</strong>{". A positive value means the weighted pattern has a positive response there. It is not yet a probability or necessarily a meaningful named feature."}</Prose>

<ConvolutionPatchLab />

<Prose>{"This is the operation deep-learning libraries usually call convolution, although the precise mathematical operation is "}<strong>{"cross-correlation"}</strong>{": the filter is used in the displayed orientation. Mathematical convolution reverses its spatial axes. For learned filters either convention can represent the same family of operations, but matching a fixed filter or an implementation requires knowing the convention. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html"}>{"PyTorch Conv2d contract"}</a>{"."}</Prose>

<Prose>{"A filter "}<InlineMath>{"[1,0,-1]"}</InlineMath>{" on a one-dimensional signal compares an earlier and a later value. A constant signal gives zero away from padding; an abrupt change gives a response. On an image, a similar pattern can respond to an edge. We do not need to hand-design every detector: training adjusts the weights to reduce the task's errors."}</Prose>

<Prose>{"Why reuse them? A dense layer connecting 64 input pixels to 512 outputs has 32,768 weights before biases. Eight 3×3 filters on a one-channel image have 72 weights, yet produce eight whole maps. The local pattern detector shares evidence across positions. Dense layers can learn correlations too; they simply do not impose these particular locality and weight-sharing constraints. Positions in overlapping patches are also correlated: they are not hundreds of independent extra training specimens."}</Prose>

<H2>{"2. Channels are several measurements at each location"}</H2>

<Prose>{"An RGB image has three channels. A later hidden representation might have eight channels corresponding to eight learned responses. A normal convolution combines a patch from "}<strong>{"every input channel"}</strong>{" to make one output channel."}</Prose>

<Prose>{"For one output at location "}<InlineMath>{"(u,v)"}</InlineMath>{", with stride one and no padding:"}</Prose>

<div className="neural-equation"><MathBlock>{"y_{o,u,v}=b_o+\\sum_{c=0}^{C_{\\rm in}-1}\n\\sum_{a=0}^{k_h-1}\\sum_{b=0}^{k_w-1}\nW_{o,c,a,b}\\,x_{c,u+a,v+b}."}</MathBlock></div>

<Prose>{"Here "}<InlineMath>{"o"}</InlineMath>{" selects an output channel, "}<InlineMath>{"c"}</InlineMath>{" selects an input channel, and "}<InlineMath>{"a,b"}</InlineMath>{" select a location inside its patch. The bias "}<InlineMath>{"b_o"}</InlineMath>{" is added once after combining the channels. Its subscript distinguishes the bias from the column index "}<InlineMath>{"b"}</InlineMath>{"."}</Prose>

<Prose>{"Suppose two single-cell channels contain 2 and 3. An output channel with weights 4 and −1, bias 1, gives "}<InlineMath>{"4(2)-3+1=6"}</InlineMath>{". Another output channel can apply different weights to the same inputs. A 1×1 convolution therefore mixes channels even though it does not combine neighboring spatial positions."}</Prose>

<ConvolutionChannelsLab />

<Prose>{"For a batch, PyTorch uses logical shape "}<InlineMath>{"[N,C,H,W]"}</InlineMath>{": specimens, channels, height, width. A weight tensor has shape "}<InlineMath>{"[C_{\\rm out},C_{\\rm in},k_h,k_w]"}</InlineMath>{" when groups=1. It has no separate batch or output-location axis because those uses share its values."}</Prose>

<Prose>{""}<strong>{"Grouped convolution"}</strong>{" restricts which input channels connect to which outputs. With two groups, each half of the outputs sees only its corresponding half of the inputs; both channel counts must be divisible by two. A depthwise convolution uses one group per input channel, optionally producing several outputs per input channel. The multiplier need not be one. A later pointwise layer can mix those separate channels. We will study this design properly after the landmark architectures."}</Prose>

<H2>{"3. How a shared filter learns"}</H2>

<Prose>{"Before fitting an image classifier, use a one-dimensional example small enough to calculate completely. Input "}<InlineMath>{"x=[1,3,2]"}</InlineMath>{", filter "}<InlineMath>{"w=[1,-1]"}</InlineMath>{", no bias, produces "}<InlineMath>{"y=[-2,1]"}</InlineMath>{". Suppose the desired output is "}<InlineMath>{"[0,0]"}</InlineMath>{", and use half the "}<strong>{"sum"}</strong>{" of squared errors:"}</Prose>

<div className="neural-equation"><MathBlock>{"L=\\tfrac12((-2)^2+1^2)=2.5."}</MathBlock></div>

<Prose>{"The output gradients are "}<InlineMath>{"[-2,1]"}</InlineMath>{". The first weight contributes to both windows, so its gradient adds both contributions:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{\\partial L}{\\partial w_0}=(-2)(1)+(1)(3)=1,\\qquad\n\\frac{\\partial L}{\\partial w_1}=(-2)(3)+(1)(2)=-4."}</MathBlock></div>

<Prose>{"Gradient descent with step size 0.1 gives "}<InlineMath>{"w'=[0.9,-0.6]"}</InlineMath>{". New outputs are "}<InlineMath>{"[-0.9,1.5]"}</InlineMath>{", and new loss is 1.53. One output became worse, but the specified total objective improved. Sharing a filter means negotiating all its uses rather than independently fixing each location."}</Prose>

<Prose>{"Gradients also accumulate where windows overlap. The middle input participates with weight −1 in the first window and weight 1 in the second:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{\\partial L}{\\partial x_1}=(-2)(-1)+(1)(1)=3."}</MathBlock></div>

<Prose>{"The full input gradient is "}<InlineMath>{"[-2,3,-1]"}</InlineMath>{". If we had used mean squared error, its denominator would scale the gradients. Weight sharing itself asks us to "}<strong>{"sum"}</strong>{" contributions; averaging comes from the chosen loss reduction."}</Prose>

<ConvolutionUpdateLab />

<Prose>{"This complete program runs the calculation:"}</Prose>

<CodeBlock language={"python"}>{"import torch\nfrom torch.nn import functional as F\n\nx = torch.tensor([1., 3., 2.], dtype=torch.float64, requires_grad=True)\nw = torch.tensor([1., -1.], dtype=torch.float64, requires_grad=True)\ny = F.conv1d(x[None, None], w[None, None]).flatten()\nloss = 0.5 * y.square().sum()\nloss.backward()\nnew_w = w.detach() - 0.1 * w.grad\nnew_y = F.conv1d(x.detach()[None, None], new_w[None, None]).flatten()\nprint(\"output:\", y.detach().tolist(), \"loss:\", loss.item())\nprint(\"weight gradient:\", w.grad.tolist(), \"input gradient:\", x.grad.tolist())\nprint(\"updated:\", new_w.tolist(), \"new loss:\", (0.5 * new_y.square().sum()).item())"}</CodeBlock>

<Prose>{"For classification, hidden convolutions feed activations such as ReLU, "}<InlineMath>{"a=\\max(0,z)"}</InlineMath>{", and a final layer produces one logit per class. Cross-entropy compares those logits with the actual digit. Backpropagation performs the same accumulation through the larger graph. There is no special second optimizer for filters."}</Prose>

<H2>{"4. Plan the geometry before stacking layers"}</H2>

<Prose>{""}<strong>{"Stride"}</strong>{" is the movement between output windows. "}<strong>{"Padding"}</strong>{" supplies values outside the image boundary. "}<strong>{"Dilation"}</strong>{" spaces the sampled positions inside a filter. A three-weight filter with dilation two touches offsets 0, 2 and 4: three learned values spanning five positions."}</Prose>

<Prose>{"For one dimension, let input size be "}<InlineMath>{"n"}</InlineMath>{", kernel size "}<InlineMath>{"k"}</InlineMath>{", dilation "}<InlineMath>{"d"}</InlineMath>{", stride "}<InlineMath>{"s"}</InlineMath>{", and left/right padding "}<InlineMath>{"p_l,p_r"}</InlineMath>{". The sampled span is "}<InlineMath>{"k_{\\rm eff}=d(k-1)+1"}</InlineMath>{". Count the window starts that fit:"}</Prose>

<div className="neural-equation"><MathBlock>{"n_{\\rm out}=\\left\\lfloor\\frac{n+p_l+p_r-k_{\\rm eff}}{s}\\right\\rfloor+1."}</MathBlock></div>

<Prose>{"Apply this independently to height and width. The floor means a leftover strip can be unused. A nonpositive result means the requested operation does not fit; it is not a valid zero-sized feature map."}</Prose>

<NeuralTable caption={"4. Plan the geometry before stacking layers"} headers={[<>{"Input"}</>,<>{"Kernel / dilation"}</>,<>{"Padding left,right"}</>,<>{"Stride"}</>,<>{"Output"}</>]} rows={[[<>{"8"}</>,<>{"3 / 1"}</>,<>{"1,1"}</>,<>{"1"}</>,<>{"8"}</>],[<>{"8"}</>,<>{"3 / 1"}</>,<>{"1,1"}</>,<>{"2"}</>,<>{"4"}</>],[<>{"8"}</>,<>{"4 / 1"}</>,<>{"1,2"}</>,<>{"1"}</>,<>{"8"}</>],[<>{"8"}</>,<>{"3 / 2"}</>,<>{"2,2"}</>,<>{"1"}</>,<>{"8"}</>],[<>{"7"}</>,<>{"3 / 1"}</>,<>{"0,0"}</>,<>{"2"}</>,<>{"3"}</>]]} />

<ConvolutionGeometryLab />

<Prose>{"“Same” describes an output-size policy, not a universal padding number. For stride one, total required padding is "}<InlineMath>{"d(k-1)"}</InlineMath>{". With an even effective kernel this may need unequal sides. A four-wide kernel can use left 1/right 2 or left 2/right 1; both preserve width but align outputs differently. In PyTorch 2.14, "}<code>{"padding=\"same\""}</code>{" supports stride one. Explicit "}<code>{"F.pad"}</code>{" handles a chosen asymmetric policy."}</Prose>

<Prose>{"Two other names describe useful boundary choices. With stride and dilation one, "}<strong>{"valid"}</strong>{" uses no padding and produces "}<InlineMath>{"n-k+1"}</InlineMath>{" outputs. "}<strong>{"Full"}</strong>{" uses "}<InlineMath>{"k-1"}</InlineMath>{" padding on each side and produces "}<InlineMath>{"n+k-1"}</InlineMath>{", including windows with only a partial overlap with the original input. Both follow the same window-count formula."}</Prose>

<Prose>{"Zero padding assumes an outside value of zero. Reflection, replication and circular padding impose different boundary conditions; choose them based on what the data means. A periodic signal can justify wrapping. An ordinary photograph does not automatically continue from its right edge to its left."}</Prose>

<Prose>{"Two stride-one 3×3 layers have a five-wide possible input region; three have seven, provided dilation is one. Intermediate nonlinearities make this a different function family from a single wider linear filter. Fewer parameters or greater expressivity depends on channel counts and the precise comparison, not just the number of layers."}</Prose>

<H2>{"5. Pooling: summarize, then notice what disappeared"}</H2>

<Prose>{"A pooling operation normally works independently in each channel. For a 2×2 window "}<InlineMath>{"\\begin{bmatrix}1&4\\\\2&3\\end{bmatrix}"}</InlineMath>{", max pooling gives 4; average pooling gives 2.5. With stride two, the next window starts two cells away."}</Prose>

<Prose>{"Max pooling asks for the strongest response in the window. Average pooling asks for its mean level. Neither is universally better, and either loses information: many different windows share the same maximum or mean. A decoder may learn a plausible reconstruction using other evidence, but that does not make pooling invertible."}</Prose>

<ConvolutionPoolingLab />

<Prose>{"For a sum of max-pool outputs on "}<InlineMath>{"[1,4,3]"}</InlineMath>{", kernel two and stride one, the outputs are "}<InlineMath>{"[4,4]"}</InlineMath>{". The input gradient is "}<InlineMath>{"[0,2,0]"}</InlineMath>{": the middle value wins both windows. At a tie the mathematical maximum has several valid subgradients; the implementation chooses an index. Our CPU fixture on "}<InlineMath>{"[2,2,1]"}</InlineMath>{" yields gradient "}<InlineMath>{"[1,1,0]"}</InlineMath>{". Do not build an argument that depends on a universal tie winner across all backends."}</Prose>

<Prose>{"Padding needs care with negative values. Max-pool padding behaves like negative infinity, so an invented zero cannot beat a real negative input. Average pooling can include or exclude padded zeros from the denominator. With "}<InlineMath>{"[-2,-3]"}</InlineMath>{", kernel three and one padded cell on each side, the two averages are both "}<InlineMath>{"-5/3"}</InlineMath>{" when padding counts, versus "}<InlineMath>{"-5/2"}</InlineMath>{" when it does not. These policies change numbers despite matching output shapes. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.MaxPool2d.html"}>{"MaxPool2d"}</a>{", "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.AvgPool2d.html"}>{"AvgPool2d"}</a>{"."}</Prose>

<Prose>{"Adaptive average pooling specifies an output size instead of one fixed stride. For input length five and output length three, its bins are indices "}<InlineMath>{"[0,2)"}</InlineMath>{", "}<InlineMath>{"[1,4)"}</InlineMath>{", "}<InlineMath>{"[3,5)"}</InlineMath>{". On "}<InlineMath>{"[1,2,3,4,5]"}</InlineMath>{", this gives "}<InlineMath>{"[1.5,3,4.5]"}</InlineMath>{". Bins can overlap; “adaptive” does not mean an exact equal disjoint partition for every size. "}<a href={"https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/AdaptivePooling.h"}>{"PyTorch's bin-boundary implementation"}</a>{"."}</Prose>

<Prose>{""}<strong>{"Global average pooling"}</strong>{" averages an entire spatial map into one value per channel. It allows a classifier head to receive the same number of features at different spatial sizes. It discards location in that final map, which can be useful or harmful depending on the task. It does not guarantee that the earlier map is unaffected by cropping or shifting."}</Prose>

<Prose>{"This connects to dropout: dropping responses before max pooling changes which value wins, and a sampled zero can exceed a negative activation. That is a particular noisy objective, not an algebraically forbidden ordering. Our comparison below omits dropout so we can inspect the pooling choice clearly."}</Prose>

<H2>{"6. Receptive fields: where can this number get information?"}</H2>

<Prose>{"A first-layer 3×3 output sees a 3×3 patch. A later output sees several earlier outputs, each with its own input region. The receptive field is built by tracing those connections backwards."}</Prose>

<Prose>{"Track three values per spatial axis:"}</Prose>

<ul><li>{""}<InlineMath>{"r"}</InlineMath>{": width of the theoretical bounding region in input coordinates."}</li><li>{""}<InlineMath>{"j"}</InlineMath>{": spacing between adjacent output centers, measured in input pixels."}</li><li>{""}<InlineMath>{"a"}</InlineMath>{": center of the first output, with input pixel centers at "}<InlineMath>{"0.5,1.5,\\ldots"}</InlineMath>{"."}</li></ul>

<Prose>{"Start with "}<InlineMath>{"r=1,j=1,a=0.5"}</InlineMath>{". A layer with "}<InlineMath>{"k,d,s,p_l"}</InlineMath>{" changes them by:"}</Prose>

<div className="neural-equation"><MathBlock>{"r'=r+d(k-1)j,\\qquad j'=sj,\\qquad\na'=a+\\left(\\frac{d(k-1)}2-p_l\\right)j."}</MathBlock></div>

<Prose>{"The old "}<InlineMath>{"j"}</InlineMath>{" belongs on the right side of all three equations. A pooling window expands the region too."}</Prose>

<Prose>{"For a 32-wide input:"}</Prose>

<NeuralTable caption={"6. Receptive fields: where can this number get information?"} headers={[<>{"Operation"}</>,<>{"Output width"}</>,<>{""}<InlineMath>{"r"}</InlineMath>{""}</>,<>{""}<InlineMath>{"j"}</InlineMath>{""}</>,<>{"First center "}<InlineMath>{"a"}</InlineMath>{""}</>]} rows={[[<>{"3-wide convolution, pad 1"}</>,<>{"32"}</>,<>{"3"}</>,<>{"1"}</>,<>{"0.5"}</>],[<>{"2-wide pooling, stride 2"}</>,<>{"16"}</>,<>{"4"}</>,<>{"2"}</>,<>{"1"}</>],[<>{"3-wide convolution, pad 1"}</>,<>{"16"}</>,<>{"8"}</>,<>{"2"}</>,<>{"1"}</>],[<>{"2-wide pooling, stride 2"}</>,<>{"8"}</>,<>{"10"}</>,<>{"4"}</>,<>{"2"}</>],[<>{"3-wide convolution, pad 1"}</>,<>{"8"}</>,<>{"18"}</>,<>{"4"}</>,<>{"2"}</>],[<>{"Average over all 8 positions"}</>,<>{"1"}</>,<>{"46"}</>,<>{"4"}</>,<>{"16"}</>]]} />

<Prose>{"A width of 46 on a 32-wide input includes padded coordinates; it does not mean 46 observed pixels or wraparound. For output index "}<InlineMath>{"u"}</InlineMath>{", center is "}<InlineMath>{"a+uj"}</InlineMath>{"; its bounding endpoints are that center plus/minus "}<InlineMath>{"(r-1)/2"}</InlineMath>{". Intersect with actual input coordinates to distinguish observed values from padding."}</Prose>

<ConvolutionReceptiveLab />

<Prose>{"The region can have holes. Two three-wide, dilation-two layers reach offsets "}<InlineMath>{"\\{-4,-2,0,2,4\\}"}</InlineMath>{": bounding width nine, only five positions. Using dilation one followed by two reaches all seven offsets from −3 to 3. A bounding width is not a count of connected pixels."}</Prose>

<Prose>{"For residual additions or concatenated branches, take the union of contributing input regions. Equal tensor shapes do not prove equal spatial alignment. Branches with different center offsets may add features referring to different image locations. The "}<a href={"https://distill.pub/2019/computing-receptive-fields/"}>{"receptive-field coordinate derivation"}</a>{" gives a useful deeper treatment of alignment."}</Prose>

<H2>{"7. Build and inspect a real digit classifier"}</H2>

<Prose>{"Our complete "}<a href={"/learn-code/convolution-pooling-receptive-fields/convolution-experiments.py"}>{"CPU experiment"}</a>{" includes the direct NumPy operation, tensor fixtures, twelve training runs and saved intermediate maps. Keep "}<a href={"/learn-code/convolution-pooling-receptive-fields/digits-400.csv"}>{"the attributed CSV"}</a>{" next to it. Setup from that folder:"}</Prose>

<Prose>{"On Windows PowerShell:"}</Prose>

<CodeBlock language={"powershell"}>{"python -m venv .venv\n.\\.venv\\Scripts\\python.exe -m pip install numpy==2.3.5 scikit-learn==1.9.1 torch==2.14.0\n.\\.venv\\Scripts\\python.exe convolution-experiments.py"}</CodeBlock>

<Prose>{"On macOS/Linux, use "}<code>{"python3 -m venv .venv"}</code>{", then replace each "}<code>{".\\.venv\\Scripts\\python.exe"}</code>{" above with "}<code>{".venv/bin/python"}</code>{". Calling the environment's interpreter directly avoids depending on shell activation."}</Prose>

<Prose>{"It needs no GPU, network-loaded model or unspecified image folder. Installation needs network access; after installation the data and experiment run offline. The author executed it on Python 3.12.14, PyTorch 2.14.0+cpu, one CPU thread."}</Prose>

<ConvolutionProgram file="convolution-experiments.py" title="Read the complete convolution and digit experiment program" />

<Prose>{"The CSV contains 400 actual UCI optical-digit images: first 40 examples of each digit from sklearn's 1,797-row copy of the historical UCI test partition. Values are integers from 0 to 16, divided by 16 using the documented measurement range. A fixed stratified split, seed 22, uses 280 rows to fit and 120 for development comparisons. This is a small educational split, not the official dataset evaluation or evidence of performance on unseen writers. "}<a href={"/learn-code/convolution-pooling-receptive-fields/data-provenance.md"}>{"Data provenance"}</a>{"."}</Prose>

<Prose>{"The main CNN is:"}</Prose>

<CodeBlock language={"text"}>{"[N,1,8,8]\n→ Conv 1→8, 3×3, pad 1 → ReLU       [N,8,8,8]\n→ Pool 2×2, stride 2                [N,8,4,4]\n→ Conv 8→16, 3×3, pad 1 → ReLU      [N,16,4,4]\n→ Pool 2×2, stride 2                [N,16,2,2]\n→ Flatten 64 values → Linear 64→10  [N,10] logits"}</CodeBlock>

<Prose>{"Convolutions provide local weighted sums, ReLU makes the composition nonlinear, pooling reduces spatial size, and the final layer combines the remaining features into class scores. The second convolution's weights have shape "}<InlineMath>{"[16,8,3,3]"}</InlineMath>{", not "}<InlineMath>{"[16,1,3,3]"}</InlineMath>{"."}</Prose>

<Prose>{"We compare four declared configurations: dense 64→32 tanh→10; CNN with max pooling; the same CNN with average pooling; max-pool CNN whose final 2×2 map is globally averaged before a smaller head. The max and average CNNs start with identical weights for each seed. The global-average model shares the initial convolution tensors but has a differently sized head; the dense baseline is a different architecture. All use Adam, learning rate 0.003, 400 full-batch updates, no augmentation, dropout or weight decay. Seeds 1, 2 and 3 were chosen in advance."}</Prose>

<NeuralTable caption={"7. Build and inspect a real digit classifier"} headers={[<>{"Model"}</>,<>{"Parameters"}</>,<>{"Seed 1: CE / correct of 120"}</>,<>{"Seed 2"}</>,<>{"Seed 3"}</>]} rows={[[<>{"Dense baseline"}</>,<>{"2,410"}</>,<>{"0.07949 / 117"}</>,<>{"0.08647 / 118"}</>,<>{"0.09342 / 116"}</>],[<>{"CNN, max pooling"}</>,<>{"1,898"}</>,<>{"0.08332 / 118"}</>,<>{"0.10153 / 116"}</>,<>{"0.05194 / 119"}</>],[<>{"CNN, average pooling"}</>,<>{"1,898"}</>,<>{"0.10358 / 116"}</>,<>{"0.11126 / 118"}</>,<>{"0.13594 / 115"}</>],[<>{"CNN, global-average head"}</>,<>{"1,418"}</>,<>{"0.08249 / 117"}</>,<>{"0.13742 / 113"}</>,<>{"0.17901 / 113"}</>]]} />

<Prose>{"These are actual final-update measurements. Every run fits all 280 training labels. The CNN uses fewer parameters than this dense baseline; it does not win every seed or metric. Accuracy counts and cross-entropy measure different aspects: a few confident mistakes can increase CE even with similar counts. The program retains intermediate train/development traces rather than drawing an imagined smooth learning curve."}</Prose>

<ConvolutionMeasuredLab />

<Prose>{"A deliberately difficult stress check shifts each development image one pixel right or down, fills the exposed edge with zero and keeps its original label. This tests a changed input condition; it can also clip meaningful strokes, so it is not a clean proof of translation behavior on an unlimited canvas."}</Prose>

<Prose>{"For seed 1, the dense model gets 57/120 right-shifted images correct, and the max-pool CNN gets 72/120. Their unshifted counts were 117 and 118. Down-shift counts are 71 and 75. Even the global-average CNN falls from 117 to 63 right-shifted. All twelve stress results are saved. Weight sharing helps organize the model, but does not remove the need to define and validate deployment variation."}</Prose>

<Prose>{""}<strong>{"Before changing the experiment"}</strong>{", write the expected effect, the single change and the metric. An augmentation trial should transform training data only; assess the predeclared development views consistently. Once their outcomes guide choices, these rows remain development data. A final performance claim needs a separate untouched evaluation protocol."}</Prose>

<H2>{"8. Optional: reach, influence and shifts"}</H2>

<Prose>{"The theoretical field says which paths exist. An input gradient "}<InlineMath>{"\\partial z/\\partial x_{u,v}"}</InlineMath>{" says how a particular scalar output changes locally around a particular input, with the current parameters. A ReLU can close a route; two contributions can cancel. A zero gradient does not erase an architectural connection."}</Prose>

<Prose>{"Our exact linear example repeatedly applies the averaging filter "}<InlineMath>{"[1,1,1]/3"}</InlineMath>{", without nonlinearities. At depth two the central output's input weights are "}<InlineMath>{"[1,2,3,2,1]/9"}</InlineMath>{". More paths reach central positions. The support width is "}<InlineMath>{"2L+1"}</InlineMath>{", while the distribution's variance is "}<InlineMath>{"2L/3"}</InlineMath>{": it behaves like a sum of "}<InlineMath>{"L"}</InlineMath>{" independent offsets chosen from −1,0,1 for this mathematical construction."}</Prose>

<ConvolutionInfluenceLab />

<Prose>{"At depth 20, support width is 41; positions with at least 1% of the peak span 21. That threshold is a chosen display rule, not a universal definition of effective receptive field. Learned filters, input-dependent gates and averaging across examples change the profile. The "}<a href={"https://arxiv.org/pdf/1701.04128"}>{"effective receptive field paper"}</a>{" develops qualified Gaussian-like behavior under assumptions and empirical settings; it does not make every individual trained gradient a Gaussian. A completely closed ReLU route should display “zero gradient” rather than normalize zero into a misleading heatmap."}</Prose>

<Prose>{"Now distinguish "}<strong>{"equivariance"}</strong>{", where shifting input shifts the output map correspondingly, from "}<strong>{"invariance"}</strong>{", where output stays identical. Stride-one correlation with circular boundaries commutes with circular shifts. Our exact fixture gives difference zero. Switching that fixture to zero-padded boundaries gives maximum difference one."}</Prose>

<Prose>{"Downsampling introduces a grid phase. Sampling every second element of "}<InlineMath>{"[1,-1,1,-1,1,-1]"}</InlineMath>{" gives "}<InlineMath>{"[1,1,1]"}</InlineMath>{"; shift the input one position first and it gives "}<InlineMath>{"[-1,-1,-1]"}</InlineMath>{". A local two-value average before sampling gives zeros in both cases. This is a simple illustration of aliasing and low-pass filtering, not a claim that blurring makes every classifier invariant."}</Prose>

<Prose>{"Max pooling can tolerate some within-window motion, yet moving a response across a window boundary moves its pooled result. Global averaging is invariant to a permutation of an already-computed map. Input shifts need not produce only a permutation of that map. "}<a href={"https://proceedings.mlr.press/v97/zhang19a.html"}>{"Zhang's anti-aliasing study"}</a>{" investigates this distinction in trained networks."}</Prose>

<ConvolutionShiftLab />

<H2>{"9. Optional: the reverse operation is a scatter, not an undo"}</H2>

<Prose>{"Our one-dimensional filter can be written as a sparse matrix:"}</Prose>

<div className="neural-equation"><MathBlock>{"C=\\begin{bmatrix}1&-1&0\\\\0&1&-1\\end{bmatrix},\\qquad y=Cx."}</MathBlock></div>

<Prose>{"The transpose sends each output-side value back along the same connections, adding where they meet. It obeys "}<InlineMath>{"\\langle Cx,g\\rangle=\\langle x,C^\\top g\\rangle"}</InlineMath>{". For "}<InlineMath>{"x=[1,3,2]"}</InlineMath>{", "}<InlineMath>{"g=[2,-3]"}</InlineMath>{", both sides are −7, and "}<InlineMath>{"C^\\top g=[2,-5,3]"}</InlineMath>{"."}</Prose>

<Prose>{"But "}<InlineMath>{"C^\\top Cx=[-2,3,-1]"}</InlineMath>{", not the original input. "}<strong>{"Transposed convolution"}</strong>{" is this structured transpose operation, not an inverse. A constant added to every element of "}<InlineMath>{"x"}</InlineMath>{" disappears under this difference filter, so exact recovery from its output alone is impossible."}</Prose>

<ConvolutionTransposeLab />

<Prose>{"Three input values equal to one, kernel "}<InlineMath>{"[1,1,1]"}</InlineMath>{", stride two, produce coverage "}<InlineMath>{"[1,1,2,1,2,1,1]"}</InlineMath>{". In two dimensions, uneven overlap can multiply into a checkerboard. A kernel divisible by stride can equalize interior coverage, yet learned weights can still make artifacts. Resizing followed by an ordinary convolution offers a different constraint; it is not an unconditional artifact cure. "}<a href={"https://distill.pub/2016/deconv-checkerboard/"}>{"Visual explanation of checkerboard artifacts"}</a>{"."}</Prose>

<Prose>{"For symmetric padding "}<InlineMath>{"p"}</InlineMath>{", transposed output size is"}</Prose>

<div className="neural-equation"><MathBlock>{"n_{\\rm out}=(n_{\\rm in}-1)s-2p+d(k-1)+{\\rm output\\_padding}+1."}</MathBlock></div>

<Prose>{"Different input sizes can map to the same strided forward size. "}<code>{"output_padding"}</code>{" selects a compatible output size; it does not mean appending that many zero-valued output cells. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.ConvTranspose2d.html"}>{"ConvTranspose2d"}</a>{"."}</Prose>

<H2>{"10. Optional: implement the same math efficiently"}</H2>

<Prose>{"The retained program's "}<code>{"direct_conv2d"}</code>{" follows the nested sums explicitly, including groups, symmetric padding, stride and dilation. Five float64 cases compare it with PyTorch, including rectangular kernels and depthwise multiplier two, with maximum absolute discrepancy below "}<InlineMath>{"10^{-12}"}</InlineMath>{". This checks this bounded CPU arithmetic; it is not a GPU benchmark."}</Prose>

<Prose>{"An alternative extracts each patch into a column. For the first image, the column matrix is"}</Prose>

<div className="neural-equation"><MathBlock>{"P=\\begin{bmatrix}\n1&2&0&1\\\\2&0&1&3\\\\0&1&2&1\\\\1&3&1&0\n\\end{bmatrix}."}</MathBlock></div>

<Prose>{"Multiplying "}<InlineMath>{"[1,-1,0,1]P"}</InlineMath>{" yields "}<InlineMath>{"[0,5,0,-2]"}</InlineMath>{", then reshape to 2×2. This is often called im2col. It exposes matrix multiplication, but explicitly copying overlapping patches can use substantial memory."}</Prose>

<Prose>{"Folding those columns back adds overlaps. The center input appears four times, an edge-middle twice, a corner once. "}<code>{"Fold(Unfold(x))"}</code>{" therefore equals an overlap-count grid times "}<code>{"x"}</code>{". Dividing by that grid recovers covered positions; an uncovered position with count zero cannot be recovered this way. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.Fold.html"}>{"Fold contract"}</a>{"."}</Prose>

<ConvolutionPatchMatrixFigure />

<Prose>{"Implicit matrix-multiplication algorithms generate patch addresses without materializing the whole expanded matrix. Transform methods such as FFT or Winograd change how the arithmetic is organized. Shape, datatype, hardware and workspace constraints affect the useful choice. These are implementation strategies for the operator, not new learned representations. "}<a href={"https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html"}>{"NVIDIA convolution algorithm guide, sections 3–4.2"}</a>{"."}</Prose>

<Prose>{"For groups "}<InlineMath>{"g"}</InlineMath>{", a "}<InlineMath>{"k_h\\times k_w"}</InlineMath>{" layer has"}</Prose>

<div className="neural-equation"><MathBlock>{"C_{\\rm out}(C_{\\rm in}/g)k_hk_w"}</MathBlock></div>

<Prose>{"weights, plus "}<InlineMath>{"C_{\\rm out}"}</InlineMath>{" biases if used. Forward multiply-accumulates per specimen are that weight count times "}<InlineMath>{"H_{\\rm out}W_{\\rm out}"}</InlineMath>{". If counting one multiplication and addition separately, use two FLOPs per MAC, state that convention, and account for other operators separately."}</Prose>

<Prose>{"Our first convolution has 72 weights, 80 parameters including biases, and "}<InlineMath>{"8\\cdot8\\cdot72=4{,}608"}</InlineMath>{" MACs. The second has 1,152 weights, 1,168 parameters and "}<InlineMath>{"4\\cdot4\\cdot1{,}152=18{,}432"}</InlineMath>{" MACs. The head adds 650 parameters and 640 MACs: total 1,898 parameters and 23,680 MACs for these affine operations. Pooling and activations have additional work. The dense baseline has 2,368 MACs despite more parameters. A parameter comparison is not a latency comparison."}</Prose>

<Prose>{"A 3×3 depthwise-plus-pointwise pair with multiplier one uses "}<InlineMath>{"9C_{\\rm in}+C_{\\rm in}C_{\\rm out}"}</InlineMath>{" weights, versus "}<InlineMath>{"9C_{\\rm in}C_{\\rm out}"}</InlineMath>{" for a dense 3×3 layer. The dense-to-separable weight ratio is "}<InlineMath>{"9C_{\\rm out}/(9+C_{\\rm out})"}</InlineMath>{", before biases. It also constrains the computation differently; fewer arithmetic operations do not guarantee proportional wall-clock savings."}</Prose>

<Prose>{"Memory format is another independent choice. PyTorch channels-last storage can preserve the logical "}<InlineMath>{"[N,C,H,W]"}</InlineMath>{" shape while changing strides in memory. It is not the same operation as permuting the tensor's logical axes. "}<a href={"https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html"}>{"Channels-last tutorial"}</a>{"."}</Prose>

<Prose>{"An evaluation-mode convolution followed by BatchNorm with fixed running statistics can be combined algebraically. For output channel "}<InlineMath>{"o"}</InlineMath>{", set"}</Prose>

<div className="neural-equation"><MathBlock>{"\\alpha_o=\\frac{\\gamma_o}{\\sqrt{v_o+\\epsilon}},\\qquad\nW'_o=\\alpha_o W_o,\\qquad b'_o=\\beta_o+\\alpha_o(b_o-\\mu_o)."}</MathBlock></div>

<Prose>{"Our float64 fixture matches within "}<InlineMath>{"3.4\\times10^{-16}"}</InlineMath>{". Training BatchNorm depends on the current batch, so that fixed folding argument does not apply. ReLU remains nonlinear even if a backend executes it in the same kernel. Memory layout changes, algebraic folding and actual kernel fusion should be evaluated separately when performance becomes the task."}</Prose>

<Prose>{"One useful connection goes beyond recognizing images. A fixed grid stencil "}<InlineMath>{"\\begin{bmatrix}0&1&0\\\\1&-4&1\\\\0&1&0\\end{bmatrix}"}</InlineMath>{" computes a discrete Laplacian numerator. With center temperature 30 and four neighbors at 20, it gives −40: local curvature toward cooler surroundings. For grid spacing "}<InlineMath>{"h"}</InlineMath>{", divide by "}<InlineMath>{"h^2"}</InlineMath>{"; a diffusion equation also needs diffusivity, a time discretization and boundary conditions. This uses the same local weighted-sum mechanism, but the weights represent a specified numerical operator rather than parameters learned from labels."}</Prose>

<H2>{"Implement the pullback and batch the arithmetic"}</H2>

<Prose>{"The direct "}<code>{"direct_conv2d"}</code>{" routine in "}<a href={"/learn-code/convolution-pooling-receptive-fields/convolution-experiments.py"}>{"convolution-experiments.py"}</a>{" is a transparent indexing oracle: every output, input group and tap is visible. Its scalar Python loops are not the final recommendation for processing images. The companion "}<a href={"/learn-code/convolution-pooling-receptive-fields/convolution_pullbacks.py"}>{"convolution_pullbacks.py"}</a>{" retains the spatial-tap loops but performs all examples, channels and output locations together using "}<code>{"einsum"}</code>{". This opens the operation without allocating a full expanded patch matrix."}</Prose>

<Prose>{"For valid dense NCHW cross-correlation, one tap contributes "}<code>{"images[:, :, row:row+OH, col:col+OW]"}</code>{" contracted with "}<code>{"weights[:, :, row, col]"}</code>{". The forward contraction is "}<code>{"bihw,oi->bohw"}</code>{". Given upstream derivative "}<code>{"bohw"}</code>{", the weight gradient contracts "}<code>{"bohw,bihw->oi"}</code>{"; the input gradient contracts "}<code>{"bohw,oi->bihw"}</code>{" and "}<strong>{"adds"}</strong>{" it into the corresponding input slice. Overlapping windows write to the same input, so assignment would lose contributions. Bias gradients sum batch and spatial axes."}</Prose>

<Prose>{"The complete program contains all forward/backward functions and a same-input "}<code>{"F.conv2d"}</code>{" comparison for every derivative. It has the deliberately stated boundary of dense, stride-one, unpadded valid convolution; the earlier general routine still owns grouped, dilated and strided address calculation. Extending this pullback means applying exactly those same addresses in reverse. Autograd can perform that composition in the real classifier, so there is no need to reimplement its engine."}</Prose>

<Prose>{"For B examples, I input channels, O outputs, K spatial taps and P output positions, the arithmetic is O(BIOKP), with activation/parameter/output storage plus a tap-sized contraction temporary; there is no explicit O(BIKP) im2col buffer. The maintained backend can use different kernels and layouts. This cost statement is not a measured speed ranking."}</Prose>

<Prose>{"The same companion implements valid one-dimensional max/average pooling and their pullbacks. "}<code>{"sliding_window_view"}</code>{" exposes windows without copying each one. Maximum routing saves the first maximizing index per window; average routing distributes an upstream value over the window. "}<code>{"np.add.at"}</code>{" accumulates when maxima or average windows share an input. This is the implementation of the overlap diagrams, not an import of an opaque pooling operation. The script checks both with native PyTorch pooling. Padding, adaptive-window geometry and two-dimensional indexing follow the explicit conventions taught earlier; the real model continues to use native pooling."}</Prose>

<Prose>{"Run "}<code>{"python convolution_pullbacks.py"}</code>{" with NumPy and PyTorch. The declared result is equality within float64 tolerances, not a new accuracy measurement. These code paths are separate from the saved digit fits."}</Prose>

<ConvolutionProgram file="convolution_pullbacks.py" title="Read the complete vectorized convolution and pooling pullbacks" />

<Prose>{""}<strong>{"Change the implementation:"}</strong>{" replace the upstream all-ones pooling vector by "}<code>{"[2,−1,3]"}</code>{" for values "}<code>{"[1,4,3,2,−1]"}</code>{", size3, stride1. Then extend the convolution pullback to stride2 by using the original forward sampling slice in both directions."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Max pooling sends each upstream value to one saved winner; mean pooling sends one third to each covered value. A stride changes selected input addresses, not the summation rule."}</Prose>

</details>

<details>

<summary>Solution and success criteria</summary>

<Prose>{"The max winners are input indices1,1,2, so the derivative is "}<code>{"[0,1,3,0,0]"}</code>{". Mean pooling gives "}<code>{"[2/3,1/3,4/3,2/3,1]"}</code>{". For strided convolution select "}<code>{"row:row+stride*OH:stride"}</code>{" and its analogous column slice; accumulate the input derivative into that same strided slice. Match forward, input, weight and bias derivatives to "}<code>{"F.conv2d(..., stride=2)"}</code>{" on a rectangular image. Equality of outputs alone does not catch a mistaken overlapping scatter."}</Prose>

</details>

<H3>{"Adaptive pooling without a hidden implementation"}</H3>

<Prose>{"The same program implements "}<code>{"adaptive_average1d"}</code>{" and its explicit pullback. For output bin "}<code>{"i"}</code>{", use "}<code>{"start = floor(i*n/m)"}</code>{" and "}<code>{"end = ceil((i+1)*n/m)"}</code>{", then average the half-open input range "}<code>{"[start, end)"}</code>{". These bins can overlap; they are not necessarily a disjoint partition. A prefix sum makes each bin sum a subtraction, for O(n+m) time and storage. The pullback adds "}<code>{"upstream[i] / (end-start)"}</code>{" to every member of that bin; a range-add difference array accumulates this in O(n+m), including overlapping bins. Prefix subtraction can lose relative precision when subtracting two large, almost equal cumulative sums; this float64 teaching implementation is not a guarantee of identical summation error to a native kernel."}</Prose>

<Prose>{"The program compares values and input gradients with "}<code>{"torch.nn.functional.adaptive_avg_pool1d"}</code>{" for 5→3, 3→5 and global 5→1 pooling. The upsampling-shaped case is intentional: adaptive average pooling can create overlapping repeated bins even though no interpolation rule is being applied. "}<strong>{"Changed-input task:"}</strong>{" pool "}<code>{"[1, 2, 3, 4, 5]"}</code>{" into three bins with output cotangent "}<code>{"[1, 2, 3]"}</code>{". The bins are "}<code>{"[0,2)"}</code>{", "}<code>{"[1,4)"}</code>{" and "}<code>{"[3,5)"}</code>{", giving values "}<code>{"[1.5, 3, 4.5]"}</code>{" and input gradient "}<code>{"[1/2, 7/6, 2/3, 13/6, 3/2]"}</code>{". Verify the middle inputs collect all their participating bins, then extend the same construction along both axes for adaptive 2-D average pooling."}</Prose>

<H2>{"11. Practice: construct, diagnose, transfer"}</H2>

<Prose>{"Try the questions before opening the solutions. A correct number without explaining which cells and parameters participated is incomplete."}</Prose>

<ol><li>{""}<strong>{"Changed filter."}</strong>{" On the first 3×3 image, replace the filter by "}<InlineMath>{"\\begin{bmatrix}1&0\\\\0&-1\\end{bmatrix}"}</InlineMath>{". Calculate all outputs. Then increase only the central pixel from 1 to 2 and identify every changed output."}</li><li>{""}<strong>{"Mix channels."}</strong>{" Two input channels have single-cell values 2 and 3. Build two output channels with a 1×1 convolution: first equals their sum, second equals their difference. Supply the complete weight and bias arrays. Which output changes if only channel two increases by one?"}</li><li>{""}<strong>{"A shape is not a coordinate."}</strong>{" Input width 10, kernel four, stride one, dilation one. Choose padding that preserves width. Give both asymmetric choices nearest to symmetry and their first output centers. Explain why equal-sized branches may still be misaligned."}</li><li>{""}<strong>{"Pooling ambiguity."}</strong>{" Give two different 2×2 windows with max four and average two. Can their pooled outputs reconstruct the original window? Then calculate the gradient of the sum of stride-one, size-two max pools on "}<InlineMath>{"[3,1,4]"}</InlineMath>{"."}</li><li>{""}<strong>{"A new receptive field."}</strong>{" Start with a 20-wide input. Apply kernel three/stride two/pad one, then kernel three/dilation two/stride one/pad two. Find output width, "}<InlineMath>{"r,j,a"}</InlineMath>{". Does a bounding width alone establish that all enclosed coordinates connect?"}</li><li>{""}<strong>{"Experiment diagnosis."}</strong>{" A model changes from 118 correct and CE 0.08 to 118 correct and CE 0.16. A colleague says nothing changed because accuracy is identical. Explain what else to inspect. Propose one controlled training change to investigate the observed shift failures, including what data can inform selection."}</li><li>{""}<strong>{"Optional adjoint problem."}</strong>{" Use "}<InlineMath>{"C"}</InlineMath>{" above and output-side values "}<InlineMath>{"[1,1]"}</InlineMath>{". Compute "}<InlineMath>{"C^\\top[1,1]"}</InlineMath>{", and name two distinct inputs with the same forward output."}</li><li>{""}<strong>{"Optional cost problem."}</strong>{" For 16 input and 32 output channels, compare weights in a dense 3×3 convolution with multiplier-one depthwise 3×3 followed by pointwise 1×1. Explain what measurement is still missing before claiming a speedup."}</li></ol>

<details><summary>Hints</summary>

<ol><li>{"Name each window by its top-left coordinate; a pixel can occupy different filter positions. 2. Each output has its own row of input-channel weights. 3. Total padding must equal "}<InlineMath>{"k-1"}</InlineMath>{"; use the center recurrence. 4. A window's sum must be eight. Overlapping gradients add. 5. Update "}<InlineMath>{"r"}</InlineMath>{" using the jump from before the layer. 6. CE depends on confidence in the true label, not just the largest logit. 7. Think about the difference filter acting on a constant. 8. Count separately before dividing; do not equate arithmetic and elapsed time."}</li></ol>

</details>

<details><summary>Worked solutions</summary>

<ol><li>{"Outputs are "}<InlineMath>{"\\begin{bmatrix}0&-1\\\\-1&1\\end{bmatrix}"}</InlineMath>{". After editing the center, they become "}<InlineMath>{"\\begin{bmatrix}-1&-1\\\\-1&2\\end{bmatrix}"}</InlineMath>{". The other windows contain that pixel but multiply it by a zero coefficient, so they stay unchanged."}</li><li>{"Weights, in output/input/spatial order, are "}<InlineMath>{"[[[[1]],[[1]]],[[[1]],[[-1]]]]"}</InlineMath>{", biases "}<InlineMath>{"[0,0]"}</InlineMath>{". Outputs are "}<InlineMath>{"[5,-1]"}</InlineMath>{". Changing the second input to four gives "}<InlineMath>{"[6,-2]"}</InlineMath>{": both change, in opposite directions."}</li><li>{"Padding "}<InlineMath>{"(1,2)"}</InlineMath>{" gives first center 1; "}<InlineMath>{"(2,1)"}</InlineMath>{" gives center 0, using input centers "}<InlineMath>{"0.5,1.5,\\ldots"}</InlineMath>{". Both output widths are ten. Their center grids differ by one input pixel, so elementwise addition would combine different locations."}</li><li>{"Examples are "}<InlineMath>{"\\begin{bmatrix}4&2\\\\1&1\\end{bmatrix}"}</InlineMath>{" and "}<InlineMath>{"\\begin{bmatrix}4&0\\\\2&2\\end{bmatrix}"}</InlineMath>{". Both summaries match although the windows differ. The one-dimensional max outputs are "}<InlineMath>{"[3,4]"}</InlineMath>{"; gradient is "}<InlineMath>{"[1,0,1]"}</InlineMath>{"."}</li><li>{"First layer produces width ten, "}<InlineMath>{"r=3,j=2,a=0.5"}</InlineMath>{". Second preserves width ten and produces "}<InlineMath>{"r=11,j=2,a=0.5"}</InlineMath>{". Bounding widths alone are insufficient in general. Here the first layer's contiguous offsets plus the second layer's spaced offsets reach "}<InlineMath>{"\\{-5,-4,-3,-1,0,1,3,4,5\\}"}</InlineMath>{", leaving holes at −2 and 2 before boundary clipping."}</li><li>{"Inspect per-row true-class probabilities, especially incorrect and nearly tied examples; equal accuracy can hide worse confidence. One controlled trial adds small label-preserving translations to training images while keeping architecture, seed protocol and learning budget fixed. Inspect whether crops remain interpretable, then compare the same declared development conditions. These comparisons cannot be relabeled as untouched final-test evidence."}</li><li>{"The transpose result is "}<InlineMath>{"[1,0,-1]"}</InlineMath>{". Inputs "}<InlineMath>{"[1,3,2]"}</InlineMath>{" and "}<InlineMath>{"[2,4,3]"}</InlineMath>{" both produce "}<InlineMath>{"[-2,1]"}</InlineMath>{", because adding a constant leaves adjacent differences unchanged."}</li><li>{"Dense: "}<InlineMath>{"9(16)(32)=4,608"}</InlineMath>{" weights. Separable: "}<InlineMath>{"9(16)+16(32)=656"}</InlineMath>{", a ratio of about 7.02, excluding biases. Actual latency needs measurement for the target input/output sizes, batch, dtype, layout, backend, device and memory behavior; predictive quality also needs validation."}</li></ol>

</details>

<Prose>{"Core readiness means you can trace one output and update, connect channels correctly, plan spatial sizes and input regions, explain information lost by pooling, and interpret the controlled digit comparison. The optional questions extend that understanding; they are not a hidden requirement to begin the next lesson."}</Prose>

<Prose>{"The next topic in this module is "}<a href={"/learn/path/full-curriculum/landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet?module=deep-learning-fundamentals"}>{"Landmark Architectures: LeNet through EfficientNet"}</a>{". It will use these operators to explain architectural choices. Depthwise/dilated designs follow that topic; the route does not skip over an unprepared lesson."}</Prose>

<H2>{"References and other ways to learn"}</H2>

<ul><li>{""}<a href={"https://arxiv.org/pdf/1603.07285"}>{"Dumoulin and Visin, A Guide to Convolution Arithmetic for Deep Learning"}</a>{", v2: a visual reference for padding, stride and transposed operations. Use sections 2–3 after the geometry ruler, section 4 after the scatter example. Its operator arithmetic is durable; library-specific policies are checked separately here."}</li><li>{""}<a href={"https://www.youtube.com/watch?v=bNb2fEVKeEo"}>{"Stanford CS231n, Lecture 5: Convolutional Neural Networks"}</a>{": an alternate spoken introduction to convolution, pooling and the transition from dense layers. The official 2017 syllabus and video description were checked; the full video was not watched for this draft. Historical architecture examples are context, not current performance recommendations."}</li><li>{""}<a href={"https://distill.pub/2019/computing-receptive-fields/"}>{"Araujo, Norris and Sim, Computing Receptive Fields"}</a>{": interactive coordinate and multi-path diagrams, useful after calculating "}<InlineMath>{"r,j,a"}</InlineMath>{". Historical model comparisons do not prove that increasing receptive field alone causes better accuracy."}</li><li>{""}<a href={"https://arxiv.org/pdf/1701.04128"}>{"Luo and colleagues, Understanding the Effective Receptive Field"}</a>{": the advanced analytical and empirical source for distinguishing possible reach from concentration of input gradients. Read the assumptions before generalizing its profile shapes."}</li><li>{""}<a href={"https://distill.pub/2016/deconv-checkerboard/"}>{"Odena, Dumoulin and Olah, Deconvolution and Checkerboard Artifacts"}</a>{": an especially useful visual alternate route through overlap patterns and decoder choices."}</li><li>{""}<a href={"https://proceedings.mlr.press/v97/zhang19a.html"}>{"Zhang, Making Convolutional Networks Shift-Invariant Again"}</a>{": primary research motivating anti-aliasing around downsampling. The abstract was reviewed; the specific experiment results here come from our retained calculation and digit program."}</li><li>{""}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html"}>{"PyTorch 2.14 Conv2d"}</a>{", "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.MaxPool2d.html"}>{"MaxPool2d"}</a>{", "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.AvgPool2d.html"}>{"AvgPool2d"}</a>{", "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.ConvTranspose2d.html"}>{"ConvTranspose2d"}</a>{", and "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.Fold.html"}>{"Fold"}</a>{": exact API policies, especially groups, padding, pooling denominators and overlap accumulation."}</li><li>{""}<a href={"https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html"}>{"Channels-last tutorial"}</a>{" and "}<a href={"https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html"}>{"NVIDIA convolution guide"}</a>{": optional engineering references for storage and algorithm choices. The NVIDIA examples include older device/software benchmarks; no timings from them are presented as our measurements."}</li><li>{""}<a href={"https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits"}>{"UCI Optical Recognition of Handwritten Digits"}</a>{": original data source, authors and license. Our subset, source IDs and exact split are documented in the adjacent provenance file."}</li></ul>
  </div>,
};
