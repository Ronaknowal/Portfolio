import { Prose, H2, H3, Code, CodeBlock } from '../../components/content';
import { Math, MathBlock as SharedMathBlock } from '../../components/content/Math.jsx';
import { ResidualTable, ResidualCorrectionLab, ResidualGradientFigure, ResidualDepthLab, ResidualOrderLab, ResidualProjectionLab, ResidualOpeningLab, ResidualEvidenceLab, ResidualPathsFigure, ResidualDenoisingFigure, ResidualEulerLab, ResidualMemoryFigure, ResidualProgram } from '../../components/lesson-labs/ResidualConnectionsLabs.jsx';
function MathBlock({children}) { return <div className="res-equation" role="region" tabIndex={0} aria-label="Equation; scroll horizontally when needed"><SharedMathBlock>{children}</SharedMathBlock></div>; }
export default {
  title: 'Residual Connections & Skip Connections: Keep a Path, Learn a Correction',
  readTime: '55–75 min + live exploration and practice',
  hasIntegratedGuide: true,
  content: () => <div className="residual-lesson">
<Prose><strong>{"Explore as you read."}</strong>{" Edit branch weights, scalar depth/gain, normalization placement, projection entries and Euler step; inspect recorded block omissions. Show correction contributions, current output/loss, both derivative paths and shape compatibility as inputs change. Display every intermediate gain and genuine before/after ablation record. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to recognize cancellation, shape mismatch and step-size instability despite the presence of an identity path."}</Prose>

<Prose>{"Suppose a network has already formed a useful description of an image. Its next block can replace that description completely—or keep it and propose a correction."}</Prose>

<Prose>{"A "}<strong>{"residual connection"}</strong>{" implements the second choice:"}</Prose>

<MathBlock>{"\\text{new representation}=\\text{current representation}+\\text{learned correction}."}</MathBlock>

<Prose>{"The extra route carries the current values around a group of operations and adds them at the end. This gives both information and learning signals a direct path through the block. It is a powerful design choice, but the correction can still cancel, amplify, or distort what the direct path carries."}</Prose>

<Prose><strong>{"First pass:"}</strong>{" trace one correction and its parameter update, follow the two backward paths, compare operation order and shapes, and run the small digit experiment. Practice 1–4 check those ideas. Scaled residuals, path expansions, architecture families, and the differential-equation connection are deeper branches; you can revisit them after the core."}</Prose>

<aside className="res-route"><p>Before you start: matrix multiplication, gradients and the preceding normalization/initialization lessons. The first five sections build the core; optional connections follow the real experiment.</p><nav aria-label="Residual connections lesson sections"><ol><li><a href="#one-block-with-numbers-you-can-follow">{"One block, with numbers you can follow"}</a></li><li><a href="#why-a-direct-path-helps-optimization">{"Why a direct path helps optimization"}</a></li><li><a href="#the-skip-cannot-promise-a-nonvanishing-total-gradient">{"The skip cannot promise a nonvanishing total gradient"}</a></li><li><a href="#operation-order-changes-what-is-preserved">{"Operation order changes what is preserved"}</a></li><li><a href="#shape-agreement-also-needs-coordinate-agreement">{"Shape agreement also needs coordinate agreement"}</a></li><li><a href="#start-near-identity-while-leaving-something-able-to-learn">{"Start near identity while leaving something able to learn"}</a></li><li><a href="#a-complete-comparison-on-real-handwritten-digits">{"A complete comparison on real handwritten digits"}</a></li><li><a href="#own-the-addition-reuse-the-layers-and-differentiation">{"Own the addition; reuse the layers and differentiation"}</a></li><li><a href="#many-routes-are-useful-but-they-are-not-independent-models">{"Many routes are useful, but they are not independent models"}</a></li><li><a href="#connections-that-solve-different-problems">{"Connections that solve different problems"}</a></li><li><a href="#what-really-costs-memory-and-computation">{"What really costs memory and computation"}</a></li><li><a href="#practice">{"Practice"}</a></li><li><a href="#another-way-to-learn">{"Another way to learn"}</a></li></ol></nav></aside>

<Prose>{"The "}<a href={"/learn/path/full-curriculum/weight-initialization-xavier-kaiming-p?module=deep-learning-fundamentals"}>{"previous initialization lesson"}</a>{" asked how transformations should start. This lesson asks how to connect them. We will use vectors and small MLPs first, so convolution and attention are not prerequisites."}</Prose>

<H2>{"One block, with numbers you can follow"}</H2>

<Prose>{"Write the input as "}<Math>{"x"}</Math>{", the correction function as "}<Math>{"F"}</Math>{", and the output as "}<Math>{"y"}</Math>{":"}</Prose>

<MathBlock>{"y=x+F(x)."}</MathBlock>

<Prose>{"The straight-through route is an "}<strong>{"identity"}</strong>{": it returns exactly the value it receives. The other route may contain several layers. Calling it a "}<strong>{"skip"}</strong>{" does not mean the computer usually skips calculating "}<Math>{"F(x)"}</Math>{"; both routes contribute to an ordinary forward pass."}</Prose>

<Prose>{"Consider a two-coordinate representation and a small linear correction:"}</Prose>

<MathBlock>{"x=\\begin{bmatrix}2\\\\-1\\end{bmatrix},\\qquad\nW=\\begin{bmatrix}0.1&0.2\\\\0&-0.5\\end{bmatrix},\\qquad F(x)=Wx."}</MathBlock>

<Prose>{"Follow the two paths:"}</Prose>

<ResidualTable caption="Forward routes in the worked vector example" headers={[<>{"Route"}</>,<>{"Calculation"}</>,<>{"Value"}</>]} rows={[[<>{"Direct"}</>,<>{"Preserve "}<Math>{"x"}</Math></>,<><Math>{"[2,-1]"}</Math></>],[<>{"Correction, coordinate 1"}</>,<><Math>{"0.1(2)+0.2(-1)"}</Math></>,<>{"0"}</>],[<>{"Correction, coordinate 2"}</>,<><Math>{"0(2)-0.5(-1)"}</Math></>,<>{"0.5"}</>],[<>{"Add coordinate by coordinate"}</>,<><Math>{"[2,-1]+[0,0.5]"}</Math></>,<><Math>{"[2,-0.5]"}</Math></>]]} />

<Prose>{"The first coordinate stayed unchanged. The second moved upward by 0.5. A correction may be positive or negative; the network should be able to remove a feature as well as add one."}</Prose>

<Prose><strong>{"Visual walkthrough:"}</strong>{" carry two labeled values along a direct lane, and send the same values through the weight matrix in another lane. Reveal the products, the correction vector, and the addition in order. Then change one weight and inspect which output coordinates change. Use the coordinate labels to explain each change."}</Prose>

<Prose>{"If the desired output is "}<Math>{"t=[1,0]"}</Math>{", this correction has not finished the job. Use half the squared distance as the loss:"}</Prose>

<MathBlock>{"\\mathcal L=\\tfrac12\\|y-t\\|^2\n=\\tfrac12(1^2+(-0.5)^2)=0.625."}</MathBlock>

<Prose>{"The derivative with respect to "}<Math>{"y"}</Math>{" is "}<Math>{"g=y-t=[1,-0.5]"}</Math>{". For "}<Math>{"F(x)=Wx"}</Math>{", the weight gradient is the outer product "}<Math>{"gx^\\top"}</Math>{":"}</Prose>

<MathBlock>{"\\nabla_W\\mathcal L=\n\\begin{bmatrix}2&-1\\\\-1&0.5\\end{bmatrix}."}</MathBlock>

<Prose>{"One gradient-descent step with learning rate 0.1 gives"}</Prose>

<MathBlock>{"W_{\\mathrm{new}}=W-0.1\\nabla_W\\mathcal L\n=\\begin{bmatrix}-0.1&0.3\\\\0.1&-0.55\\end{bmatrix}."}</MathBlock>

<Prose>{"The new correction is "}<Math>{"[-0.5,0.75]"}</Math>{", the new output is "}<Math>{"[1.5,-0.25]"}</Math>{", and the loss is 0.15625. The skip path did not learn a weight. It changed the input-output function and therefore the error used to train the correction."}</Prose>

<Prose>{"This is a linear example for tracing arithmetic, not a claim that one linear residual layer is more expressive than a general linear layer. Since "}<Math>{"x+Wx=(I+W)x"}</Math>{", both can represent the same linear maps. A practical residual branch usually includes a nonlinearity."}</Prose>

<ResidualCorrectionLab />

<H2>{"Why a direct path helps optimization"}</H2>

<Prose>{"A deeper model can have many possible representations, but a training algorithm still has to find useful parameters. The historical "}<strong>{"degradation problem"}</strong>{" was an observation that adding layers to certain plain networks increased their training error. A worse training fit cannot be explained simply by saying the larger model memorized its training data too well."}</Prose>

<Prose>{"If extra blocks can represent identity maps, a deeper model can reproduce a shallower model's function. That establishes the existence of an equally good configuration, not a guarantee that an optimizer will reach it. The original ResNet work proposed learning "}<Math>{"H(x)-x"}</Math>{" when the desired block mapping is "}<Math>{"H(x)"}</Math>{", making the identity case "}<Math>{"F(x)=0"}</Math>{" convenient to represent. "}<a href={"https://arxiv.org/pdf/1512.03385"}>{"He et al., §§1 and 3"}</a></Prose>

<Prose>{"Several qualifications matter. The identity-based argument requires the added blocks to support that identity construction. A nonlinearity after the addition can change that construction. Zero correction is an available setting, not the default of every randomly initialized block. And a training-error plot alone cannot identify every cause of an optimization difficulty; it does not rule out all gradient-flow problems."}</Prose>

<Prose>{"The direct route also changes backpropagation. An addition node sends the upstream derivative into both inputs. At the block input the two contributions meet:"}</Prose>

<MathBlock>{"\\nabla_x\\mathcal L=g+J_F(x)^\\top g,"}</MathBlock>

<Prose>{"where "}<Math>{"J_F(x)"}</Math>{" is the matrix describing small output changes caused by small input changes. We use column gradients, which is why the transpose appears."}</Prose>

<Prose>{"In the numerical example, the direct route contributes "}<Math>{"[1,-0.5]"}</Math>{". The correction route contributes "}<Math>{"W^\\top g=[0.1,0.45]"}</Math>{". Their total is "}<Math>{"[1.1,-0.05]"}</Math>{"."}</Prose>

<Prose>{"Notice the second coordinate: its gradient became "}<strong>{"smaller"}</strong>{" after the two routes combined. An unattenuated contribution does not imply that the final sum has a lower bound on its magnitude. This is the distinction between seeing a path in a graph and evaluating the complete derivative."}</Prose>

<ResidualGradientFigure />

<H2>{"The skip cannot promise a nonvanishing total gradient"}</H2>

<Prose>{"Use a scalar correction "}<Math>{"F(x)=ax"}</Math>{". The block is "}<Math>{"y=(1+a)x"}</Math>{", with derivative "}<Math>{"1+a"}</Math>{". Ten identical blocks have derivative "}<Math>{"(1+a)^{10}"}</Math>{":"}</Prose>

<ResidualTable caption="Scalar derivatives through ten residual blocks" headers={[<>{"Branch slope "}<Math>{"a"}</Math></>,<>{"One-block derivative"}</>,<>{"Ten-block derivative"}</>]} rows={[[<>{"−1"}</>,<>{"0"}</>,<>{"0"}</>],[<>{"−0.5"}</>,<>{"0.5"}</>,<>{"0.0009765625"}</>],[<>{"0"}</>,<>{"1"}</>,<>{"1"}</>],[<>{"0.1"}</>,<>{"1.1"}</>,<>{"2.59374"}</>],[<>{"1"}</>,<>{"2"}</>,<>{"1024"}</>]]} />

<Prose>{"At "}<Math>{"a=-1"}</Math>{", the learned branch cancels the identity exactly. At "}<Math>{"a=-0.5"}</Math>{", sensitivity contracts repeatedly despite a skip in every block. At "}<Math>{"a=1"}</Math>{", it explodes. At "}<Math>{"a=0"}</Math>{", the route is an exact identity."}</Prose>

<Prose><strong>{"Gradient investigation:"}</strong>{" choose the correction slope and a number of blocks, then inspect contraction, preservation, or growth while displaying the computed gain. Build a stack that still contains every skip but reduces the ten-block gain below 0.001. Then repair that gain by editing the correction. This is a constructive way to understand both the benefit and the limit."}</Prose>

<ResidualDepthLab />

<Prose>{"For vector blocks "}<Math>{"x_{k+1}=x_k+F_k(x_k)"}</Math>{", the full Jacobian is the ordered product"}</Prose>

<MathBlock>{"J_{\\mathrm{total}}=(I+J_{F_{L-1}})\\cdots(I+J_{F_0})."}</MathBlock>

<Prose>{"The identity term offers a direct contribution. Other terms can reinforce or cancel it, and matrices generally cannot be reordered. If each residual Jacobian has operator norm at most "}<Math>{"\\epsilon<1"}</Math>{", one-block directional gains lie between "}<Math>{"1-\\epsilon"}</Math>{" and "}<Math>{"1+\\epsilon"}</Math>{". Across "}<Math>{"L"}</Math>{" blocks, even the bound "}<Math>{"(1-\\epsilon)^L"}</Math>{" can become small. Small per-block changes and depth must be considered together."}</Prose>

<Prose>{"The identity-mappings study analyzes these direct routes and compares shortcut and activation choices. Its empirical evidence supports their usefulness; it should not be turned into a universal impossibility of vanishing gradients. "}<a href={"https://arxiv.org/pdf/1603.05027"}>{"He et al., §§2–4"}</a></Prose>

<H2>{"Operation order changes what is preserved"}</H2>

<Prose>{"Compare three formulas:"}</Prose>

<ResidualTable caption="Operation placement with a zero correction" headers={[<>{"Pattern"}</>,<>{"Output"}</>,<>{"What happens when "}<Math>{"F=0"}</Math>{"?"}</>]} rows={[[<>{"Pure additive residual"}</>,<><Math>{"x+F(x)"}</Math></>,<>{"Exactly "}<Math>{"x"}</Math></>],[<>{"ReLU after addition"}</>,<><Math>{"\\operatorname{ReLU}(x+F(x))"}</Math></>,<><Math>{"\\operatorname{ReLU}(x)"}</Math></>],[<>{"LayerNorm after addition"}</>,<><Math>{"\\operatorname{LN}(x+F(x))"}</Math></>,<><Math>{"\\operatorname{LN}(x)"}</Math></>]]} />

<Prose>{"For "}<Math>{"x=[-2,1]"}</Math>{", zero correction gives "}<Math>{"[-2,1]"}</Math>{" in the first row and "}<Math>{"[0,1]"}</Math>{" in the second. The second row's local Jacobian is "}<Math>{"\\operatorname{diag}(0,1)"}</Math>{". Its negative coordinate cannot pass through unchanged."}</Prose>

<Prose>{"LayerNorm across the two features produces approximately "}<Math>{"[-0.999998,0.999998]"}</Math>{" using epsilon "}<Math>{"10^{-5}"}</Math>{", unit affine scale and zero bias. Adding the same constant to both input coordinates leaves LayerNorm's output unchanged. Its Jacobian therefore removes that common-shift direction, even though the output magnitude looks well controlled."}</Prose>

<Prose>{"In "}<strong>{"pre-activation"}</strong>{" residual designs, normalization and activation live inside the correction path, and the addition has no trailing activation. In "}<strong>{"pre-norm"}</strong>{" designs, a common form is "}<Math>{"x+F(\\operatorname{LN}(x))"}</Math>{". With zero correction, both preserve the raw skip value. The normalization still affects the branch and its derivative."}</Prose>

<ResidualOrderLab />

<Prose>{"Moving an operation is not just relabeling a block. A post-activation ResNet can work well; Torchvision's standard BasicBlock and Bottleneck still apply ReLU after addition. The correct conclusion is to understand the chosen design, not to treat every post-activation block as a bug. "}<a href={"https://docs.pytorch.org/vision/0.26/_modules/torchvision/models/resnet.html"}>{"Torchvision 0.26 ResNet source"}</a></Prose>

<Prose>{"For a future Transformer lesson, the two patterns are:"}</Prose>

<MathBlock>{"\\begin{aligned}\n\\text{pre-norm: }&u=x+\\operatorname{Attention}(\\operatorname{LN}(x)),\\\\\n&y=u+\\operatorname{MLP}(\\operatorname{LN}(u));\\\\\n\\text{post-norm: }&u=\\operatorname{LN}(x+\\operatorname{Attention}(x)),\\\\\n&y=\\operatorname{LN}(u+\\operatorname{MLP}(u)).\n\\end{aligned}"}</MathBlock>

<Prose>{"Treat Attention here as another same-shape transformation; its mechanism comes later. Pre-norm keeps normalization off the direct route. It does not force the accumulated residual stream to have constant variance, or eliminate every need for warmup. Xiong et al. study expected initialization gradients under a particular model and demonstrate useful pre-norm training results; that is narrower than stability at any depth and any learning rate. "}<a href={"https://arxiv.org/abs/2002.04745"}>{"On Layer Normalization in the Transformer Architecture"}</a></Prose>

<H2>{"Shape agreement also needs coordinate agreement"}</H2>

<Prose>{"Addition combines corresponding entries. For a batch of vectors, a residual output with shape [batch, 32] should be added to another [batch, 32] representation with the intended feature correspondence. A [batch, 1] output may broadcast without a runtime error while performing a quite different operation."}</Prose>

<Prose>{"When dimensions change, the skip can use a projection "}<Math>{"P"}</Math>{":"}</Prose>

<MathBlock>{"y=P x+F(x)."}</MathBlock>

<Prose>{"For example,"}</Prose>

<MathBlock>{"P=\\begin{bmatrix}1&0\\\\0&1\\\\1&1\\end{bmatrix}\n\\quad\\text{maps}\\quad [2,-1]\\ \\text{to}\\ [2,-1,1]."}</MathBlock>

<Prose>{"The correction must now contain three coordinates. This is no longer an identity shortcut. An upstream gradient "}<Math>{"[1,2,3]"}</Math>{" returns through the skip as "}<Math>{"P^\\top[1,2,3]=[4,5]"}</Math>{". A projection changes both forward representation and backward sensitivity."}</Prose>

<Prose>{"For an image tensor [batch, channels, height, width], a future convolutional block may change channels and spatial resolution. A 1×1 convolution can mix channels at each position; stride can select a coarser grid. Both branches must agree on output shape "}<strong>{"and spatial alignment"}</strong>{", including padding choices. A projection is one solution; pooling or deliberate padding may fit other designs. Do not insert a learned projection where an identity already expresses the intended connection without considering its effects."}</Prose>

<Prose><strong>{"Shape investigation:"}</strong>{" connect named feature sockets, choose identity or an explicit projection, and compute the resulting values. Check the feature labels and actual repeated values as well as the dimension count."}</Prose>

<ResidualProjectionLab />

<H2>{"Start near identity while leaving something able to learn"}</H2>

<Prose>{"There are different ways to begin with a small correction."}</Prose>

<Prose>{"A two-layer branch can use "}<Math>{"F(x)=W_2\\phi(W_1x)"}</Math>{", with random "}<Math>{"W_1"}</Math>{" and zero "}<Math>{"W_2"}</Math>{". The initial correction is zero. The first gradient into "}<Math>{"W_1"}</Math>{" is zero, but "}<Math>{"W_2"}</Math>{" can receive a gradient from the nonzero hidden features. After "}<Math>{"W_2"}</Math>{" moves, the earlier layer can start learning."}</Prose>

<Prose>{"Compare this with "}<Math>{"F(x)=\\operatorname{ReLU}(W_2\\phi(W_1x))"}</Math>{", again with "}<Math>{"W_2=0"}</Math>{". Using PyTorch's zero derivative for ReLU at zero blocks the gradient into "}<Math>{"W_2"}</Math>{" as well. A graph that passes the input through beautifully may have a correction branch that never learns."}</Prose>

<Prose>{"The saved fixture uses "}<Math>{"x=[1,2]"}</Math>{", "}<Math>{"W_1=I"}</Math>{", "}<Math>{"W_2=0"}</Math>{", and loss "}<Math>{"\\tfrac12\\|y\\|^2"}</Math>{". With the final linear output, "}<Math>{"\\nabla_{W_2}\\mathcal L=[[1,2],[2,4]]"}</Math>{". With the extra final ReLU, both weight gradients are zero."}</Prose>

<Prose><strong>{"ReZero"}</strong>{" instead uses a trainable scalar:"}</Prose>

<MathBlock>{"y=x+\\alpha F(x),\\qquad \\alpha_{\\mathrm{initial}}=0."}</MathBlock>

<Prose>{"At initialization, the block's input-output map is identity. For upstream gradient "}<Math>{"g"}</Math>{","}</Prose>

<MathBlock>{"\\frac{\\partial\\mathcal L}{\\partial\\alpha}=g^\\top F(x),\n\\qquad\n\\nabla_W\\mathcal L=\\alpha J_{F,W}^\\top g."}</MathBlock>

<Prose>{"The scalar may move before the branch parameters do. It is a weight, not a probability, and may become negative. With "}<Math>{"x=[1,2]"}</Math>{", "}<Math>{"F(x)=[0,0.7]"}</Math>{", target zero, and half squared loss, the scalar gradient is 1.4. SGD at 0.1 moves "}<Math>{"\\alpha"}</Math>{" from zero to −0.14. A later update can reach the branch weights. If the scalar gradient is also zero, that first movement is not guaranteed. "}<a href={"https://arxiv.org/pdf/2003.04887"}>{"ReZero paper, mechanism and scalar example"}</a></Prose>

<Prose><strong>{"LayerScale"}</strong>{" uses a separate learned scalar for each output feature: "}<Math>{"x+\\lambda\\odot F(x)"}</Math>{". This gives channels separate update scales. Small nonzero values can allow small branch gradients immediately. Zero per-channel values remain per-channel parameters; they do not lose that flexibility just because their initial values coincide. The CaiT study tested particular small initial values and training recipes, not a universal “all networks above 24 layers need "}<Math>{"10^{-6}"}</Math>{"” law. "}<a href={"https://arxiv.org/pdf/2103.17239"}>{"LayerScale mechanism and initialization study"}</a></Prose>

<Prose>{"A fixed factor such as "}<Math>{"1/\\sqrt L"}</Math>{" is another possible design experiment. Its rationale often assumes approximately comparable residual contributions. In general,"}</Prose>

<MathBlock>{"E\\|x+F(x)\\|^2=E\\|x\\|^2+E\\|F(x)\\|^2+2E[x^\\top F(x)],"}</MathBlock>

<Prose>{"where the expectations average over the input distribution under study. Correlations can change the growth. Scaling, initialization, normalization, and optimizer choices work together; none can be inferred from depth alone."}</Prose>

<ResidualOpeningLab />

<H2>{"A complete comparison on real handwritten digits"}</H2>

<Prose>{"Download "}<a href={"/learn-assets/residual-connections/residual-experiments.py"}>{"residual-experiments.py"}</a>{" and "}<a href={"/learn-assets/residual-connections/digits-400.csv"}>{"digits-400.csv"}</a>{" into one directory. The "}<a href={"/learn-assets/residual-connections/data-provenance.md"}>{"provenance"}</a>{" gives attribution and the exact split. These 400 real 8×8 UCI digit images use pixel values 0–16. Divide by 16, flatten to 64 inputs, and classify digits 0–9."}</Prose>

<Prose>{"With PyTorch, NumPy, and scikit-learn installed:"}</Prose>

<CodeBlock language="text">{"python residual-experiments.py"}</CodeBlock>

<Prose>{"The program contains all data loading, initialization, models, training, diagnostics, and evaluation. It was executed on CPU with PyTorch 2.14.0+cpu, NumPy 2.3.5 and scikit-learn 1.9.1. It downloads no model or data."}</Prose>

<ResidualProgram />

<Prose>{"Every model starts with a 64→32 linear stem and tanh, and ends with a 32→10 logit head. The stem-only baseline connects those two directly. Other models insert 2, 6, or 12 blocks, each containing"}</Prose>

<MathBlock>{"F(h)=W_2\\tanh(W_1\\operatorname{LN}(h)+b_1)+b_2."}</MathBlock>

<Prose>{"LayerNorm operates across each row's 32 features with its standard affine parameters. The final branch output is linear, so its correction may have either sign."}</Prose>

<ResidualTable caption="Four block return expressions" headers={[<>{"Mode"}</>,<>{"Block output"}</>]} rows={[[<>{"Plain"}</>,<><Math>{"F(h)"}</Math></>],[<>{"Residual"}</>,<><Math>{"h+F(h)"}</Math></>],[<>{"Fixed scaled"}</>,<><Math>{"h+F(h)/\\sqrt L"}</Math></>],[<>{"Learned zero gate"}</>,<><Math>{"h+\\alpha F(h)"}</Math>{", each block's "}<Math>{"\\alpha"}</Math>{" starts at zero"}</>]]} />

<Prose>{"The learned-gate model retains LayerNorm to keep this comparison matched. It is a ReZero-style gate experiment, not a reproduction of the paper's complete normalization-free recipe."}</Prose>

<Prose>{"Read "}<Code>{"Refinement.forward"}</Code>{": the important architectural difference is the return expression. "}<Code>{"DigitNetwork"}</Code>{" constructs the same branch weights for a given seed/block index and the same stem/head for a given seed, across modes and depths. All models use the same stratified 280/120 training/validation split. Each trains for 250 full-batch Adam updates at learning rate 0.003. The three seeds were declared before examining results."}</Prose>

<Prose>{"Each ordinary block has 2,176 trainable parameters: two 32×32 weights and two 32 biases, plus 32 LayerNorm scales and 32 offsets. The stem/head have 2,410 total. Thus a 12-block plain, residual, or fixed-scaled model has 28,522 parameters; the learned-gate model has 12 extra scalars. Parameter equality does not imply equal runtime, and deeper configurations do more work per update."}</Prose>

<Prose>{"Actual final seed-1 results:"}</Prose>

<ResidualTable caption="Selected actual seed-1 final results" headers={[<>{"Blocks"}</>,<>{"Mode"}</>,<>{"Training CE"}</>,<>{"Validation CE"}</>,<>{"Correct / 120"}</>]} rows={[[<>{"0"}</>,<>{"Stem only"}</>,<>{"0.027306"}</>,<>{"0.107264"}</>,<>{"117"}</>],[<>{"2"}</>,<>{"Plain"}</>,<>{"0.000968"}</>,<>{"0.195945"}</>,<>{"115"}</>],[<>{"2"}</>,<>{"Residual"}</>,<>{"0.001076"}</>,<>{"0.112114"}</>,<>{"115"}</>],[<>{"6"}</>,<>{"Plain"}</>,<>{"0.000702"}</>,<>{"0.254232"}</>,<>{"116"}</>],[<>{"6"}</>,<>{"Residual"}</>,<>{"0.000664"}</>,<>{"0.221809"}</>,<>{"116"}</>],[<>{"12"}</>,<>{"Plain"}</>,<>{"0.002000"}</>,<>{"0.599136"}</>,<>{"110"}</>],[<>{"12"}</>,<>{"Residual"}</>,<>{"0.000389"}</>,<>{"0.303736"}</>,<>{"111"}</>],[<>{"12"}</>,<>{"Fixed scaled"}</>,<>{"0.000405"}</>,<>{"0.144360"}</>,<>{"113"}</>],[<>{"12"}</>,<>{"Learned zero gate"}</>,<>{"0.000184"}</>,<>{"0.161761"}</>,<>{"116"}</>]]} />

<Prose>{"These results do "}<strong>{"not"}</strong>{" reproduce the exact historical degradation experiment. All these plain networks fit the training set well at this budget. Their deeper validation results can still worsen. Across seeds, the 12-block residual model scores 111–116 correct; the scaled model 113–116; and the learned-gate model 115–117. The much smaller stem-only baseline scores 117–118. More depth is not necessary for every dataset."}</Prose>

<Prose>{"The program records CE and correct counts at updates 0, 1, 25, 100, and 250. It also records layer activation mean squares and gradients of the loss on a fixed 32-example training subset. Those diagnostics answer a specific local question; a larger gradient norm is not automatically more useful. Parameter-displacement records confirm that branches actually changed during fitting."}</Prose>

<Prose><strong>{"Try a controlled change:"}</strong>{" first choose one seed and compare plain versus residual at the same depth. Then compare residual versus scaled. Identify which variables are held fixed and which change. For a new experiment, alter one branch placement or initialization, explain the observed effect, and keep the resulting record separate. These are validation comparisons; this packet has no final test estimate."}</Prose>

<ResidualEvidenceLab />

<H2>{"Own the addition; reuse the layers and differentiation"}</H2>

<Prose>{"The scratch operation here is the routing equation, not a replacement for every linear layer. In "}<a href={"/learn-assets/residual-connections/residual-experiments.py"}>{"residual-experiments.py"}</a>{", "}<Code>{"Refinement.forward"}</Code>{" computes a correction, checks its exact shape and returns "}<Code>{"inputs + scale * correction"}</Code>{". "}<Code>{"scale"}</Code>{" is a buffer for a fixed choice and an "}<Code>{"nn.Parameter"}</Code>{" for ReZero. That distinction determines whether it is saved as model state and whether an optimizer updates it. The complete "}<Code>{"DigitNetwork"}</Code>{" and fitting loop are also the ordinary PyTorch route; PyTorch has no mandatory opaque “residual layer” that a researcher must import."}</Prose>

<CodeBlock language="python">{"    def forward(self, inputs):\n        correction = self.upper(torch.tanh(self.lower(self.norm(inputs))))\n        if self.mode == \"plain\":\n            return correction\n        if correction.shape != inputs.shape:\n            raise ValueError(\"residual addition requires the same shape and coordinate meaning\")\n        return inputs + self.scale*correction"}</CodeBlock>

<Prose>The branch normalizes only its own input, creates hidden features, and produces a signed correction with a final linear map. The plain case returns that correction directly. The residual case first rejects accidental broadcasting, then adds the original input. This shape check cannot inspect feature meaning; preserving coordinate semantics remains part of the architecture design. There is no special residual operator hidden behind this return statement.</Prose>

<Prose>{"The function "}<Code>{"mechanisms"}</Code>{" pairs the hand-derived input/weight derivatives with "}<Code>{"torch.autograd.grad"}</Code>{", traces a zero last layer and a zero gate separately, and checks that addition itself saves no values for its derivative. Reuse the implemented "}<a href={"/learn/path/full-curriculum/backpropagation-automatic-differentiation"}>{"Backpropagation lesson"}</a>{" for the differentiation engine, and "}<a href={"/learn/path/full-curriculum/batch-layer-group-rms-normalization"}>{"Normalization"}</a>{" for the "}<Code>{"LayerNorm"}</Code>{" primitive. These are actual implementation owners. The preceding initialization lesson supplies the scaling discussion; the local calculations here expose how that scaling changes a connected block."}</Prose>

<Prose>In the fitting loop, <Code>model.train()</Code> selects training behavior, <Code>optimizer.zero_grad(set_to_none=True)</Code> clears the previous derivative, and cross-entropy receives logits rather than already-normalized probabilities. After checking that loss is finite, <Code>loss.backward()</Code> traverses both routes and <Code>optimizer.step()</Code> updates the parameters. Evaluation uses <Code>model.eval()</Code> with <Code>torch.no_grad()</Code>; these control different things. Fixed scales are saved buffers, while trainable gates enter the optimizer through <Code>model.parameters()</Code>. Match these details before attributing a changed result to the shortcut. The <a href="https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html">Module API</a> documents mode, buffer and parameter behavior.</Prose>

<Prose>{"For a concrete code modification, replace the scalar gate with a width-sized "}<Code>{"nn.Parameter(torch.zeros(width))"}</Code>{" and leave the return expression unchanged. Broadcasting then applies one gate per feature. For upstream gradient g and branch output F, each new gate gradient is "}<Code>{"g[j] * F[j]"}</Code>{", summed over batch rows if there are several. A scalar gate instead sums across features too. Check the changed gradient using the same input, target and weights, then perform one SGD step and confirm that both gates can move differently. The residual addition costs O(Bd) work and adds no d×d matrix; the branch usually dominates arithmetic. This changes the model class, so a trained scalar checkpoint cannot be called equivalent merely by repeating its scale across channels."}</Prose>

<details><summary>{"Extension hint"}</summary>

<Prose>{"Use the existing "}<Code>{"gates"}</Code>{" fixture and retain the branch while changing only the gate's shape."}</Prose>

</details>

<details><summary>{"Worked extension"}</summary>

<Prose>{"For input [1,2], branch [0,0.7], zero gate, and half squared error to zero, the vector gate gradient is [0,1.4]. At rate 0.1 it becomes [0,−0.14], whereas the scalar gate becomes −0.14 for both coordinates. The first coordinate happens to have zero correction in this fixture; edit the first branch row to make the representational difference observable. Success requires matching the separate gate gradients and subsequent outputs, not just a nonzero loss decrease."}</Prose>

</details>

<H2>{"Many routes are useful, but they are not independent models"}</H2>

<Prose>{"For two linear residual blocks, expansion is exact:"}</Prose>

<MathBlock>{"(I+W_1)(I+W_0)x=x+W_0x+W_1x+W_1W_0x."}</MathBlock>

<Prose>{"You can identify four paths. With "}<Math>{"L"}</Math>{" linear blocks there are "}<Math>{"2^L"}</Math>{" such products. They share weights and are summed, not independently trained and averaged."}</Prose>

<Prose>{"For nonlinear blocks, do not distribute a function over addition. With "}<Math>{"F_0(x)=F_1(x)=x^2"}</Math>{" and "}<Math>{"x=1"}</Math>{", the actual result is"}</Prose>

<MathBlock>{"x_1=1+1=2,\\qquad x_2=2+2^2=6."}</MathBlock>

<Prose>{"The tempting expression "}<Math>{"x+F_0(x)+F_1(x)+F_1(F_0(x))"}</Math>{" gives 4 instead. The valid telescoping form is "}<Math>{"x_L=x_0+\\sum_kF_k(x_k)"}</Math>{"; each "}<Math>{"x_k"}</Math>{" already depends on earlier corrections."}</Prose>

<Prose>{"The ensemble-of-paths interpretation is useful for thinking about multiple gradient routes and testing robustness, but it is not a theorem that deleting any block is harmless. "}<a href={"https://arxiv.org/abs/1605.06431"}>{"Veit et al."}</a></Prose>

<ResidualPathsFigure />

<Prose>{"Our program performs "}<strong>{"ablations"}</strong>{": after fitting, omit one residual block at a time and measure the changed validation result without retraining. For seed 1 with six unscaled residual blocks, the full model gets 116/120 correct. Single-block omissions range from 28 to 114 correct. That is a direct counterexample to promising deletion robustness from architecture alone. Different scaled/gated runs can be less sensitive, but those outcomes also depend on training."}</Prose>

<Prose>{"These ablations inspect the fitted model. If you use them to choose a pruned architecture, validation information has guided a new selection; a later final evaluation must account for that workflow."}</Prose>

<H2>{"Connections that solve different problems"}</H2>

<Prose><strong>{"Addition versus concatenation."}</strong>{" Addition mixes aligned coordinates into one vector. Concatenation keeps them in separate positions. If "}<Math>{"x=[2,-1]"}</Math>{" and "}<Math>{"F(x)=[0,0.5]"}</Math>{", addition yields two values "}<Math>{"[2,-0.5]"}</Math>{"; concatenation yields four "}<Math>{"[2,-1,0,0.5]"}</Math>{"."}</Prose>

<Prose>{"DenseNet supplies each layer with earlier feature maps by concatenation. Starting with "}<Math>{"c_0"}</Math>{" channels and adding "}<Math>{"k"}</Math>{" channels per layer gives "}<Math>{"c_0+Lk"}</Math>{" channels after "}<Math>{"L"}</Math>{" additions of features. Later layers receive more inputs, so stored activations and arithmetic must be assessed with the actual architecture. It is not automatically a more expensive or more accurate substitute at every budget. "}<a href={"https://arxiv.org/abs/1608.06993"}>{"DenseNet"}</a></Prose>

<Prose><strong>{"Gated carry and parallel corrections."}</strong>{" A Highway-style block can compute "}<Math>{"T(x)\\odot H(x)+(1-T(x))\\odot x"}</Math>{", with a learned gate choosing how much each feature travels along each route. The gate also changes derivatives; a nearly closed carry route is not an identity. ResNeXt instead aggregates several related transformations inside a residual branch. Its "}<strong>{"cardinality"}</strong>{" counts those transformations. Grouped convolution is a later implementation tool for such structured channel computations. "}<a href={"https://arxiv.org/abs/1505.00387"}>{"Highway Networks"}</a>{", "}<a href={"https://arxiv.org/abs/1611.05431"}>{"ResNeXt"}</a></Prose>

<Prose><strong>{"Different spatial scales inside a block."}</strong>{" Res2Net organizes smaller channel groups with hierarchical residual-like connections, allowing different groups to accumulate different spatial contexts. Revisit its diagram after learning receptive fields; for now, recognize that an outer residual connection and the internal design of its correction are separate choices. "}<a href={"https://arxiv.org/abs/1904.01169"}>{"Res2Net"}</a></Prose>

<Prose><strong>{"Remove noise by learning what to subtract."}</strong>{" If a noisy observation is "}<Math>{"z=s+n"}</Math>{", a network can estimate noise "}<Math>{"\\hat n(z)"}</Math>{" and return "}<Math>{"z-\\hat n(z)"}</Math>{". For an illustrative three-value signal, "}<Math>{"z=[0.2,0.9,0.4]"}</Math>{" and estimated noise "}<Math>{"[0.1,-0.1,0.2]"}</Math>{" give "}<Math>{"[0.1,1.0,0.2]"}</Math>{". That arithmetic does not prove the estimated noise was correct; clean/noisy examples are needed to train and evaluate the estimator. DnCNN uses residual learning of this kind. It illustrates a residual target at the whole-model level, which need not mean every internal block has a ResNet shortcut. "}<a href={"https://arxiv.org/pdf/1608.03981"}>{"DnCNN, §III-B"}</a></Prose>

<ResidualDenoisingFigure />

<Prose><strong>{"Refinement as a time step."}</strong>{" An update "}<Math>{"x_{k+1}=x_k+\\Delta t\\,f(x_k)"}</Math>{" resembles the forward-Euler method for a differential equation. For "}<Math>{"f(x)=-x"}</Math>{", it becomes "}<Math>{"x_{k+1}=(1-\\Delta t)x_k"}</Math>{". A step of 0.1 contracts by 0.9; a step of 2.5 alternates sign and grows by 1.5 per step. The differential equation decays smoothly, yet an overly large discrete step is unstable. This offers another reason that an identity route alone cannot ensure stability. Neural ODEs build a deeper connection by learning a derivative and using a differential-equation solver; an arbitrary residual network is not automatically a convergent solver. "}<a href={"https://arxiv.org/abs/1806.07366"}>{"Neural ODEs"}</a></Prose>

<ResidualEulerLab />

<H2>{"What really costs memory and computation"}</H2>

<Prose>{"An identity shortcut has no learned weights. Adding two equal-sized tensors performs one addition per entry and requires access to both values at that moment. The shortcut does not require a physical copy just because a diagram shows two lanes."}</Prose>

<Prose>{"Backward through a pure addition only needs to route the upstream derivative; it does not need to save the input values to differentiate the addition itself. Other operations in "}<Math>{"F"}</Math>{", their parameter gradients, and the lifetime of values across the branch do require storage. Memory therefore depends on the actual saved tensors and execution schedule, not simply “one saved input per plus sign.”"}</Prose>

<ResidualMemoryFigure />

<Prose>{"Activation checkpointing trades some saved intermediates for recomputation. In a simplified equal-cost chain, grouping "}<Math>{"L"}</Math>{" layers into chunks of size "}<Math>{"k"}</Math>{" suggests storage on the order of "}<Math>{"L/k+k"}</Math>{", minimized near "}<Math>{"k=\\sqrt L"}</Math>{". Real networks have unequal tensor sizes, workspaces, parameter/optimizer storage, and implementation choices. The simple expression is not a guaranteed whole-device memory bound."}</Prose>

<Prose>{"PyTorch's current checkpoint documentation recommends an explicit "}<Code>{"use_reentrant=False"}</Code>{" and warns that recomputation must agree with the original forward behavior. Random masks and mutable state deserve particular care. "}<a href={"https://docs.pytorch.org/docs/2.14/checkpoint.html"}>{"PyTorch checkpoint contract"}</a></Prose>

<Prose>{"The "}<a href={"/learn/path/full-curriculum/dropout-droppath-stochastic-depth?module=deep-learning-fundamentals"}>{"next lesson"}</a>{" will randomly mask residual branches. If code first computes "}<Code>{"F(x)"}</Code>{" and then multiplies it by zero, that branch's forward work has already happened. Avoid translating a masked-path diagram into an invented speedup."}</Prose>

<H2>{"Practice"}</H2>

<H3>{"1. Change the correction"}</H3>

<Prose>{"For the opening "}<Math>{"x,W,t"}</Math>{", change only "}<Math>{"W_{11}"}</Math>{" from 0.1 to −0.1, keeping the other three entries fixed. Calculate output and loss before looking at the solution."}</Prose>

<details><summary>{"Hint"}</summary>

<Prose>{"only the first correction coordinate changes."}</Prose>

</details>

<details><summary>{"Worked solution"}</summary>

<Prose>{"the correction becomes "}<Math>{"[-0.4,0.5]"}</Math>{", output "}<Math>{"[1.6,-0.5]"}</Math>{", and loss "}<Math>{"\\tfrac12(0.6^2+(-0.5)^2)=0.305"}</Math>{". This improves the initial 0.625, but differs from updating all weights with the gradient."}</Prose>

</details>

<H3>{"2. A skip in every block, almost no sensitivity"}</H3>

<Prose>{"Ten blocks use "}<Math>{"F(x)=-0.5x"}</Math>{". Find total derivative. Then choose a branch slope that gives total derivative exactly 1."}</Prose>

<details><summary>{"Hint"}</summary>

<Prose>{"Multiply the local derivatives. A final gain can conceal sign changes along the way."}</Prose>

</details>

<details><summary>{"Worked solution"}</summary>

<Prose><Math>{"0.5^{10}=0.0009765625"}</Math>{". Slope 0 gives each block derivative 1, hence total 1. Slope −2 also gives total 1 for an even ten blocks but reverses sign at each block; it is a useful reminder that a final gain alone does not describe intermediate behavior."}</Prose>

</details>

<H3>{"3. A projection is not identity"}</H3>

<Prose>{"Use the 3×2 projection above with input "}<Math>{"[1,3]"}</Math>{" and upstream gradient "}<Math>{"[2,-1,4]"}</Math>{". Find skip output and skip contribution to input gradient."}</Prose>

<details><summary>{"Hint"}</summary>

<Prose>{"Apply the projection in the forward direction and its transpose to the upstream gradient, coordinate by coordinate."}</Prose>

</details>

<details><summary>{"Worked solution"}</summary>

<Prose>{"output "}<Math>{"[1,3,4]"}</Math>{"; gradient "}<Math>{"[2+4,-1+4]=[6,3]"}</Math>{". The two-dimensional input receives contributions from three output coordinates."}</Prose>

</details>

<H3>{"4. Find the branch that cannot begin learning"}</H3>

<Prose>{"Both proposed blocks initially output their input. One ends its zero-initialized correction with a linear map; the other adds a ReLU after that map. Which parameter gradient should you inspect to distinguish them?"}</Prose>

<details><summary>{"Hint"}</summary>

<Prose>{"Identical forward outputs can still put a different final operation on the backward path."}</Prose>

</details>

<details><summary>{"Worked solution"}</summary>

<Prose>{"inspect the final correction weight's gradient under the same input and loss. In our fixture the linear case gets [[1,2],[2,4]], while the extra-ReLU case gets zero. Check later displacement too; the identity output by itself does not show that a correction will learn."}</Prose>

</details>

<H3>{"5. Count features"}</H3>

<Prose>{"A dense concatenation group starts with 16 channels and adds 8 channels in each of four layers. How many channels does the fourth layer receive, and how many exist afterward? Contrast addition with a fixed 16-channel residual stream."}</Prose>

<details><summary>{"Hint"}</summary>

<Prose>{"Count the outputs of preceding layers before including the fourth layer itself."}</Prose>

</details>

<details><summary>{"Worked solution"}</summary>

<Prose>{"the fourth layer receives "}<Math>{"16+3(8)=40"}</Math>{" channels; afterward there are 48. A shape-preserving additive stream remains 16 channels. Its correction must output 16, even if it uses a different internal width."}</Prose>

</details>

<H3>{"6. Interpret a disappointing ablation"}</H3>

<Prose>{"A residual classifier loses accuracy when you remove a block. Does this refute the useful direct-gradient route, prove that the block must never be pruned, or justify another experiment?"}</Prose>

<details><summary>{"Hint"}</summary>

<Prose>{"Separate an algebraic property of the graph from the measured intervention on these trained parameters."}</Prose>

</details>

<details><summary>{"Worked solution"}</summary>

<Prose>{"it shows that this trained model depends on the block under the measured intervention. The derivative identity remains true. Retraining or a different regularization method may change the outcome, but that is another experiment with its own development and final-evaluation protocol. The ablation alone cannot guarantee successful pruning."}</Prose>

</details>

<H3>{"7. An optional stability calculation"}</H3>

<Prose>{"For "}<Math>{"x_{k+1}=(1-\\Delta t)x_k"}</Math>{", find positive step sizes that strictly contract magnitude. Explain what happens at "}<Math>{"\\Delta t=2"}</Math>{"."}</Prose>

<details><summary>{"Hint"}</summary>

<Prose>{"require "}<Math>{"|1-\\Delta t|<1"}</Math>{"."}</Prose>

</details>

<details><summary>{"Worked solution"}</summary>

<Prose><Math>{"0<\\Delta t<2"}</Math>{". At 2, each step changes the sign without shrinking magnitude. For larger positive steps, magnitude grows. This is a scalar discretization calculation, not a bound for every trained residual network."}</Prose>

</details>

<H2>{"Another way to learn"}</H2>

<ul><li><a href={"https://d2l.ai/chapter_convolutional-modern/resnet.html"}>{"Dive into Deep Learning: ResNet and ResNeXt"}</a>{" offers diagrams and a convolutional implementation. Read its function-class argument after the local arithmetic; revisit its image model after the convolution lesson. Operation-count reductions should not be read as measured wall-clock speedups."}</li><li><a href={"https://arxiv.org/pdf/1603.05027"}>{"Identity Mappings in Deep Residual Networks"}</a>{" is the detailed route through the two path conditions and activation-placement experiments. Distinguish its empirical observations from the exact algebra."}</li><li><a href={"https://docs.pytorch.org/vision/0.26/_modules/torchvision/models/resnet.html"}>{"Torchvision's ResNet source"}</a>{" lets you identify actual addition, projection, and post-activation lines in a maintained implementation."}</li><li><a href={"https://arxiv.org/pdf/2003.04887"}>{"ReZero"}</a>{" and "}<a href={"https://arxiv.org/pdf/2103.17239"}>{"LayerScale"}</a>{" are useful next readings for the optional learned-scale branch."}</li></ul>

<Prose>{"The next module topic is "}<a href={"/learn/path/full-curriculum/dropout-droppath-stochastic-depth?module=deep-learning-fundamentals"}>{"Dropout, DropPath & Stochastic Depth"}</a>{". You now know exactly where a residual correction sits. The next question is what the model learns when some contributions are deliberately removed during training."}</Prose>
</div>,
};
