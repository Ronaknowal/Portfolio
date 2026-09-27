// Generated from the complete prepared manuscript by scripts/generate-spectral-lesson.mjs.
import { Prose,H2,H3,CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { CriticDerivativeFigure, SpectralCompositionFigure, PenaltyLocationsFigure, BatchGradientFigure, SpectralGroupSortFigure, SpectralOdeFigure } from '../../components/lesson-labs/SpectralMechanismFigures.jsx';
import { SpectralSlopeLab, SpectralCircleFigure, SpectralMatrixLab, SpectralDerivativeLab, SpectralConvolutionLab, SpectralPenaltyLab, SpectralLinearPenaltyLab, SpectralLibraryLab, SpectralMarginLab } from '../../components/lesson-labs/SpectralMechanismLabs.jsx';
import { SpectralMeasuredLab,SpectralProgram } from '../../components/lesson-labs/SpectralMeasuredLab.jsx';
export default {title:"Spectral Normalization & Gradient Penalty",readTime:"~80 min read + investigations and practice",hasIntegratedGuide:true,content:()=> <div className="neural-lesson neural-lesson-neutral spectral-lesson"><LessonIntro prerequisites="Matrix multiplication, vector lengths and derivatives; the loss and derivative routes are developed here." sections={[["1-a-critic-that-gives-directions","1. A critic that gives directions"],["2-what-a-sensitivity-limit-says","2. What a sensitivity limit says"],["3-spectral-normalization-control-the-strongest-stretch","3. Spectral normalization: control the strongest stretch"],["4-a-convolution-is-larger-than-its-stored-kernel","4. A convolution is larger than its stored kernel"],["5-gradient-penalty-measure-the-function-where-it-is-sampled","5. Gradient penalty: measure the function where it is sampled"],["6-implement-the-mechanism-without-losing-its-derivatives","6. Implement the mechanism without losing its derivatives"],["7-a-complete-experiment-on-recorded-digit-measurements","7. A complete experiment on recorded digit measurements"],["8-useful-connections-beyond-this-gan","8. Useful connections beyond this GAN"],["9-practice-and-transfer","9. Practice and transfer"],["references-and-another-way-to-learn","References and another way to learn"]]}>Shape a learning signal by controlling what can stretch and by measuring where it changes.</LessonIntro>
<Prose opening="exploration">{""}<strong>{"Explore as you read."}</strong>{" Edit matrix entries, normalization method, power-iteration steps, critic/input values, interpolation points and margin geometry. Update singular stretch, effective matrix, derivative paths, sampled gradient penalties and local decision distances live. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose or diagnose a constraint by what it actually bounds and where it was evaluated; a sampled penalty is not a global guarantee."}</Prose>

<Prose>{"A useful learning signal must respond to a meaningful change in the input. If it reacts enormously to an almost invisible change, optimization can become erratic. If it barely reacts to anything, it cannot tell another model how to improve. This lesson studies two ways to shape that sensitivity: rescale the transformations inside a network, or penalize its measured input gradients."}</Prose>

<Prose>{"The preceding "}<a href={"/learn/path/full-curriculum/boltzmann-machines-restricted-boltzmann-machines-rbm?module=deep-learning-fundamentals"}>{"Boltzmann Machines & Restricted Boltzmann Machines"}</a>{" lesson assigned probability through energy and a normalizing constant. Here a generator produces samples directly, and a second network supplies a learning signal by comparing generated and recorded examples. That second network makes sensitivity a practical concern."}</Prose>

<Prose opening="route">{""}<strong>{"First pass:"}</strong>{" read §§1–6 and the experiment interpretation, then attempt practice 1–6. You need vector lengths, matrix multiplication and the idea that a derivative measures change; these are refreshed below. The normalization derivative, robustness certificate, GroupSort and continuous-time connection are deeper branches. You can inspect the recorded experiment before running its complete program."}</Prose>

<Prose>{"By the end, you should be able to explain what each method controls, implement the two training updates without reversing their signs, recognize a misleading “1-Lipschitz” claim, and assess generated samples separately from critic regularization."}</Prose>

<H2>{"1. A critic that gives directions"}</H2>

<Prose>{"Imagine a generator producing two measurements of a handwritten digit: average ink on its left half and on its right half. Its input is a short random vector "}<InlineMath>{"z"}</InlineMath>{"; its output "}<InlineMath>{"G_\\theta(z)"}</InlineMath>{" is a proposed measurement pair. The parameters "}<InlineMath>{"\\theta"}</InlineMath>{" determine which pairs it tends to produce. A "}<strong>{"critic"}</strong>{" "}<InlineMath>{"f_\\phi(x)"}</InlineMath>{" gives a scalar score to a pair "}<InlineMath>{"x"}</InlineMath>{". Its parameters "}<InlineMath>{"\\phi"}</InlineMath>{" are trained to give higher average scores to recorded pairs than to generated pairs."}</Prose>

<Prose>{"The generator then changes its parameters to raise the critic's scores on its generated pairs. It does not need a target pair for each random input. It needs a direction through the differentiable chain"}</Prose>

<div className="neural-equation"><MathBlock>{"\\theta\\longrightarrow G_\\theta(z)\\longrightarrow f_\\phi(G_\\theta(z))."}</MathBlock></div>

<CriticDerivativeFigure />

<Prose>{"For the Wasserstein form used here, gradient-descent software minimizes"}</Prose>

<div className="neural-equation"><MathBlock>{"L_D=\\mathbb E[f_\\phi(G_\\theta(z))]-\\mathbb E[f_\\phi(x)]+R(\\phi),\n\\qquad L_G=-\\mathbb E[f_\\phi(G_\\theta(z))]."}</MathBlock></div>

<Prose>{""}<InlineMath>{"R"}</InlineMath>{" is a critic regularizer when one is used. The minus sign in "}<InlineMath>{"L_G"}</InlineMath>{" matters: minimizing a negative score raises the score. During a critic update, generated inputs are detached from the generator's derivative graph. During a generator update, freeze the critic's parameters but retain derivatives with respect to its input. Detaching the critic's output would remove the generator's learning signal."}</Prose>

<Prose>{"Take a one-dimensional example: recorded data are at zero, the generator emits "}<InlineMath>{"\\theta=2"}</InlineMath>{", and the fixed critic is "}<InlineMath>{"f(x)=-x"}</InlineMath>{". Recorded score is zero and generated score is −2. The generator loss is "}<InlineMath>{"\\theta"}</InlineMath>{", its derivative is one, and a step of size .1 moves the output to 1.9, toward the data. Reversing the loss sign moves it to 2.1."}</Prose>

<Prose>{"Why constrain the critic at all? If any score difference is useful, multiplying all scores by a million appears better to its maximization objective. A limit on sensitivity gives score differences a meaningful scale."}</Prose>

<H3>{"Transport distance supplies that scale"}</H3>

<Prose>{"The "}<strong>{"Wasserstein-1 distance"}</strong>{" asks for the least cost of moving probability mass from one distribution to another. Moving mass "}<InlineMath>{"a"}</InlineMath>{" through distance "}<InlineMath>{"d"}</InlineMath>{" costs "}<InlineMath>{"ad"}</InlineMath>{". For point masses at zero and "}<InlineMath>{"\\theta"}</InlineMath>{", the answer is "}<InlineMath>{"|\\theta|"}</InlineMath>{": the distance decreases continuously as the generated point approaches the recorded point."}</Prose>

<Prose>{"Under the appropriate transport assumptions, the same quantity can be written as the largest recorded-minus-generated score difference over all 1-Lipschitz critics. On Euclidean space, finite first moments ensure the usual distance is finite. Continuity with respect to generator parameters needs corresponding regularity; it is not a statement about every arbitrary parameterized distribution. A finite trained network searches a restricted family for a limited number of updates, so its observed score difference is not automatically the exact transport distance. "}<a href={"https://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf"}>{"WGAN, §2–3"}</a>{""}</Prose>

<Prose>{"For comparison, the ideal Jensen–Shannon divergence between two different point masses stays at "}<InlineMath>{"\\log 2"}</InlineMath>{", then becomes zero when they coincide. That explains one difficulty with a particular idealized objective. It does not prove that every practical GAN has zero generator gradient: finite critics and the widely used non-saturating generator loss change the argument. "}<a href={"https://proceedings.neurips.cc/paper_files/paper/2017/file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf"}>{"WGAN-GP, §2.1–2.2"}</a>{""}</Prose>

<H2>{"2. What a sensitivity limit says"}</H2>

<Prose>{"A function is "}<strong>{""}<InlineMath>{"L"}</InlineMath>{"-Lipschitz"}</strong>{", for specified input and output norms on a specified domain, when"}</Prose>

<div className="neural-equation"><MathBlock>{"\\|f(x)-f(y)\\|\\le L\\|x-y\\|\\quad\\text{for every allowed }x,y."}</MathBlock></div>

<Prose>{"If the input moves .02 units and "}<InlineMath>{"L=3"}</InlineMath>{", the output moves at most .06 units. The bound need not be attained. A constant function is 1-Lipschitz as well as 0-Lipschitz: “1-Lipschitz” means an upper bound of one, not that every slope is exactly one."}</Prose>

<Prose>{"For a continuously differentiable scalar function on a convex region, a gradient norm bounded by "}<InlineMath>{"L"}</InlineMath>{" everywhere gives the corresponding Lipschitz bound. Integrate the directional derivative along the segment from "}<InlineMath>{"x"}</InlineMath>{" to "}<InlineMath>{"y"}</InlineMath>{". Each small output change is bounded by "}<InlineMath>{"L"}</InlineMath>{" times the small input movement, so the accumulated change has the same bound. The same reasoning applies to ordinary continuous piecewise-linear networks by integrating along their pieces, including across corners. Checking a few sample gradients, however, does not establish the premise everywhere."}</Prose>

<Prose>{"Corners are allowed. "}<InlineMath>{"|x|"}</InlineMath>{" and ReLU are 1-Lipschitz even though their ordinary derivative does not exist at zero. The concern is bounded change, not whether a graph has a sharp-looking corner."}</Prose>

<SpectralSlopeLab />

<H3>{"From layers to a whole network"}</H3>

<Prose>{"For Euclidean norms, an affine layer "}<InlineMath>{"Wx+b"}</InlineMath>{" has Lipschitz constant "}<InlineMath>{"\\|W\\|_2"}</InlineMath>{", the matrix's largest singular value. The bias cancels in differences. For composition, multiply valid layer bounds. ReLU and tanh have bounds one; sigmoid has bound one-quarter; leaky ReLU with negative slope "}<InlineMath>{"\\alpha"}</InlineMath>{" has bound "}<InlineMath>{"\\max(1,|\\alpha|)"}</InlineMath>{". GELU does not have a global bound of one."}</Prose>

<Prose>{"This gives a useful ledger for a computation graph:"}</Prose>

<NeuralTable caption={"From layers to a whole network"} headers={[<>{"Construction"}</>,<>{"Valid bound when the component bounds apply"}</>]} rows={[[<>{"Composition "}<InlineMath>{"g(f(x))"}</InlineMath>{""}</>,<>{""}<InlineMath>{"L_gL_f"}</InlineMath>{""}</>],[<>{"Sum "}<InlineMath>{"f(x)+g(x)"}</InlineMath>{""}</>,<>{""}<InlineMath>{"L_f+L_g"}</InlineMath>{""}</>],[<>{"Residual block "}<InlineMath>{"x+g(x)"}</InlineMath>{""}</>,<>{""}<InlineMath>{"1+L_g"}</InlineMath>{""}</>],[<>{"Concatenation "}<InlineMath>{"(f(x),g(x))"}</InlineMath>{", Euclidean output"}</>,<>{""}<InlineMath>{"\\sqrt{L_f^2+L_g^2}"}</InlineMath>{""}</>],[<>{"Scalar multiplication "}<InlineMath>{"af(x)"}</InlineMath>{""}</>,<>{""}<InlineMath>{"|a|L_f"}</InlineMath>{""}</>]]} />

<Prose>{"A residual branch with bound .5 can produce a block with bound 1.5; the example "}<InlineMath>{"g(x)=.5x"}</InlineMath>{" attains it. LayerNorm, learned gains, pooling and attention also belong in this accounting. Normalizing only dense weights does not certify every other operation in the graph."}</Prose>

<Prose>{"Bounds can be loose. Compose "}<InlineMath>{"A=\\operatorname{diag}(3,1/3)"}</InlineMath>{" with "}<InlineMath>{"B=\\operatorname{diag}(1/3,3)"}</InlineMath>{". The product-of-norms bound is nine, while "}<InlineMath>{"BA=I"}</InlineMath>{" has norm one. The direction stretched by the first map is contracted by the second. A small sampled derivative and a large valid upper bound therefore need not contradict each other."}</Prose>

<SpectralCompositionFigure />

<H2>{"3. Spectral normalization: control the strongest stretch"}</H2>

<Prose>{"Feed every unit-length vector in two dimensions through "}<InlineMath>{"W=\\operatorname{diag}(3,1)"}</InlineMath>{". The unit circle becomes an ellipse with semi-axes three and one. A vector along the first axis is stretched threefold; one along the second is unchanged. The "}<strong>{"spectral norm"}</strong>{" is the largest stretch:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\sigma_1(W)=\\max_{\\|v\\|_2=1}\\|Wv\\|_2."}</MathBlock></div>

<Prose>{"For nonzero "}<InlineMath>{"W"}</InlineMath>{", exact unit spectral normalization uses"}</Prose>

<div className="neural-equation"><MathBlock>{"\\overline W=W/\\sigma_1(W)."}</MathBlock></div>

<Prose>{"Our ellipse now has semi-axes one and one-third. All singular values are divided by the same number. This does not make the map orthogonal, force every singular value to one, or change its rank. A target scale "}<InlineMath>{"c>0"}</InlineMath>{" instead uses "}<InlineMath>{"cW/\\sigma_1(W)"}</InlineMath>{"."}</Prose>

<SpectralCircleFigure />

<Prose>{"Normalizing to norm one can enlarge a matrix whose norm is already below one. If the intended operation is only to cap the norm, use "}<InlineMath>{"W/\\max(1,\\sigma_1(W))"}</InlineMath>{". Singular-value clipping is another operation: decompose "}<InlineMath>{"W=U\\Sigma V^T"}</InlineMath>{", cap individual singular values, then reconstruct. Entrywise weight clipping changes matrix coefficients directly and has yet another effect. Its threshold does not set a network's Lipschitz constant to that same threshold."}</Prose>

<Prose>{"The original spectral-normalization method uses a cheaper estimate of the strongest stretch during training. Its paper also studies other GAN losses, so spectral normalization is not tied exclusively to the Wasserstein objective. "}<a href={"https://arxiv.org/pdf/1802.05957"}>{"Miyato et al., §2 and Appendix A"}</a>{""}</Prose>

<H3>{"Power iteration finds a direction as well as a number"}</H3>

<Prose>{"For a nonzero left-side vector "}<InlineMath>{"u"}</InlineMath>{", repeat"}</Prose>

<div className="neural-equation"><MathBlock>{"v\\leftarrow\\frac{W^Tu}{\\|W^Tu\\|_2},\\qquad\nu\\leftarrow\\frac{Wv}{\\|Wv\\|_2},\\qquad\n\\widehat\\sigma=u^TWv."}</MathBlock></div>

<Prose>{"Multiplication amplifies components associated with larger singular values. Repeated normalization prevents the vector itself from growing without bound. With a suitable starting component, the process approaches a leading singular direction. Convergence depends on the spectral gap; a starting vector exactly orthogonal to the leading subspace can miss it."}</Prose>

<Prose>{"For "}<InlineMath>{"\\operatorname{diag}(3,1)"}</InlineMath>{", start with "}<InlineMath>{"u=(1,1)/\\sqrt2"}</InlineMath>{". One round produces "}<InlineMath>{"\\widehat\\sigma=2.863564"}</InlineMath>{". Dividing by that estimate leaves a true norm of "}<InlineMath>{"3/2.863564=1.047645"}</InlineMath>{", slightly above one. Starting with "}<InlineMath>{"u=(0,1)"}</InlineMath>{" instead yields an estimate of one forever in exact arithmetic, leaving true normalized norm three. The numerical trace makes both cases explicit."}</Prose>

<Prose>{"Training commonly retains the previous vectors because weights often move incrementally. A cached vector can be useful; it is not a universal accuracy guarantee. Near-equal leading singular values slow convergence, and a changed matrix can invalidate a previously good direction. For a dense "}<InlineMath>{"m\\times n"}</InlineMath>{" matrix, one round costs "}<InlineMath>{"O(mn)"}</InlineMath>{" arithmetic with "}<InlineMath>{"O(m+n)"}</InlineMath>{" vector storage beyond the weights. Calling the arithmetic "}<InlineMath>{"O(m+n)"}</InlineMath>{" confuses storage with work."}</Prose>

<SpectralMatrixLab />

<H3>{"Deeper: why the normalization stays in the derivative graph"}</H3>

<Prose>{"Let "}<InlineMath>{"H=\\partial L/\\partial\\overline W"}</InlineMath>{", and assume a unique positive leading singular value with unit singular vectors "}<InlineMath>{"u,v"}</InlineMath>{". Since "}<InlineMath>{"d\\sigma_1=\\langle uv^T,dW\\rangle"}</InlineMath>{", differentiating the quotient gives"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{\\partial L}{\\partial W}\n=\\frac{1}{\\sigma_1}\\left(H-\\langle H,\\overline W\\rangle uv^T\\right)."}</MathBlock></div>

<Prose>{"The second term accounts for how changing "}<InlineMath>{"W"}</InlineMath>{" also changes its scale. Treating the entire denominator as a detached constant loses that term. A practical approximation estimates "}<InlineMath>{"u,v"}</InlineMath>{" without differentiating through their iterative search, but computes "}<InlineMath>{"u^TWv"}</InlineMath>{" with "}<InlineMath>{"W"}</InlineMath>{" still in the graph. At a repeated leading singular value, the usual unique-vector derivative needs nonsmooth treatment; the displayed formula assumes uniqueness."}</Prose>

<Prose>{"The companion calculation checks the derivative for "}<InlineMath>{"W=[[2,1],[0,1]]"}</InlineMath>{" against central differences, with maximum discrepancy below "}<InlineMath>{"7\\times10^{-12}"}</InlineMath>{". The derivation explains the term; numerical agreement is a useful check on this particular implementation, not a proof for all matrices."}</Prose>

<SpectralDerivativeLab /><SpectralProgram filename="sensitivity-calculations.py" />

<H2>{"4. A convolution is larger than its stored kernel"}</H2>

<Prose>{"A kernel is reused at many spatial locations. Flattening its stored coefficients into a matrix does not generally produce the matrix that maps an entire image to its entire output."}</Prose>

<Prose>{"For input "}<InlineMath>{"(x_1,x_2,x_3)"}</InlineMath>{" and valid stride-one kernel "}<InlineMath>{"[1,1]"}</InlineMath>{","}</Prose>

<div className="neural-equation"><MathBlock>{"y=(x_1+x_2,x_2+x_3),\\qquad\nA=\\begin{bmatrix}1&1&0\\\\0&1&1\\end{bmatrix}."}</MathBlock></div>

<Prose>{"The stored row kernel has norm "}<InlineMath>{"\\sqrt2"}</InlineMath>{". The full operator has norm "}<InlineMath>{"\\sqrt3"}</InlineMath>{", because the shared middle input contributes to both outputs. Dividing the kernel by "}<InlineMath>{"\\sqrt2"}</InlineMath>{" therefore leaves a full-operator norm of "}<InlineMath>{"\\sqrt{3/2}\\approx1.224745"}</InlineMath>{"."}</Prose>

<Prose>{"Now use four inputs and stride two: the two windows do not overlap. The full operator consists of two disjoint copies of that row kernel; dividing by "}<InlineMath>{"\\sqrt2"}</InlineMath>{" gives norm one. A four-position circular stride-one convolution has full norm two before normalization, leaving "}<InlineMath>{"\\sqrt2"}</InlineMath>{" after the same kernel rescaling. Padding, stride, spatial size and overlap are part of the operator definition."}</Prose>

<SpectralConvolutionLab />

<Prose>{"This does not make kernel spectral normalization useless. It changes how strongly weights can act and is widely studied as a regularizer. It does mean that a certificate about the whole convolution requires an appropriate operator bound or computation. Fourier-based exact results for circular convolutions have their own boundary assumptions; they cannot silently be applied to every zero-padded convolution. "}<a href={"https://arxiv.org/pdf/1805.10408"}>{"Sedghi et al., operator analysis"}</a>{""}</Prose>

<H2>{"5. Gradient penalty: measure the function where it is sampled"}</H2>

<Prose>{"The WGAN gradient penalty draws a recorded input "}<InlineMath>{"x"}</InlineMath>{", a generated input "}<InlineMath>{"\\widetilde x"}</InlineMath>{", and "}<InlineMath>{"\\epsilon\\sim U[0,1]"}</InlineMath>{", then forms"}</Prose>

<div className="neural-equation"><MathBlock>{"\\widehat x=\\epsilon x+(1-\\epsilon)\\widetilde x,\\qquad\nR_{GP}=\\lambda\\mathbb E\\left[(\\|\\nabla_{\\widehat x}f(\\widehat x)\\|_2-1)^2\\right]."}</MathBlock></div>

<Prose>{"It asks a direct question about the complete critic: how sensitive is its score at this interpolated input? The derivative is with respect to input coordinates, not the critic's parameters. Training then differentiates the penalty with respect to parameters, which requires a derivative graph through that first derivative."}</Prose>

<Prose>{"Why target one? Under the transport theorem's conditions, an optimal critic has unit directional slope along relevant transport segments. Actual WGAN-GP samples random recorded/generated pairs, not a solved optimal transport coupling. The method uses that theory as motivation for a practical sampled regularizer. Its finite samples and soft penalty do not impose a global hard constraint. "}<a href={"https://proceedings.neurips.cc/paper_files/paper/2017/file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf"}>{"Gulrajani et al., Proposition 1 and §4"}</a>{""}</Prose>

<H3>{"Work through a penalty update"}</H3>

<Prose>{"For "}<InlineMath>{"f_w(x)=w^Tx"}</InlineMath>{", the input gradient is simply "}<InlineMath>{"w"}</InlineMath>{". Let "}<InlineMath>{"w=(3,4)"}</InlineMath>{", whose Euclidean norm is five, and let "}<InlineMath>{"\\lambda=2"}</InlineMath>{". The penalty is "}<InlineMath>{"2(5-1)^2=32"}</InlineMath>{". For nonzero "}<InlineMath>{"w"}</InlineMath>{", its parameter derivative is"}</Prose>

<div className="neural-equation"><MathBlock>{"\\nabla_wR=2\\lambda(\\|w\\|_2-1)\\frac{w}{\\|w\\|_2}=(9.6,12.8)."}</MathBlock></div>

<Prose>{"A penalty-only descent step of size .1 gives "}<InlineMath>{"w'=(2.04,2.72)"}</InlineMath>{", norm 3.4 and penalty 11.52. It moved toward norm one without jumping directly there. In GAN training, the adversarial-loss gradient is added to this gradient; the result need not decrease the penalty every step."}</Prose>

<SpectralLinearPenaltyLab />

<Prose>{"Two-sided target-one penalty, an upper-bound penalty and a zero-centered penalty prefer different functions:"}</Prose>

<NeuralTable caption={"Work through a penalty update"} headers={[<>{"Gradient norm "}<InlineMath>{"r"}</InlineMath>{""}</>,<>{""}<InlineMath>{"(r-1)^2"}</InlineMath>{""}</>,<>{""}<InlineMath>{"\\max(0,r-1)^2"}</InlineMath>{""}</>,<>{""}<InlineMath>{"r^2"}</InlineMath>{""}</>]} rows={[[<>{"0"}</>,<>{"1"}</>,<>{"0"}</>,<>{"0"}</>],[<>{".5"}</>,<>{".25"}</>,<>{"0"}</>,<>{".25"}</>],[<>{"1"}</>,<>{"0"}</>,<>{"0"}</>,<>{"1"}</>],[<>{"2"}</>,<>{"1"}</>,<>{"1"}</>,<>{"4"}</>]]} />

<Prose>{"The first column penalizes a constant function even though it satisfies the 1-Lipschitz upper bound. The second does not penalize slopes below one. The third favors a zero gradient at the sampled locations. Those are different objectives, not alternative spellings of the same constraint."}</Prose>

<H3>{"R1 and R2 change both the target and sampling location"}</H3>

<Prose>{"R1 uses "}<InlineMath>{"\\frac\\gamma2\\mathbb E_{x\\sim p_{data}}\\|\\nabla_x f(x)\\|^2"}</InlineMath>{"; R2 uses the corresponding expectation on generated inputs. They are zero-centered penalties at different distributions. Moving WGAN-GP's target-one penalty onto real inputs alone does not turn it into R1. Their convergence analysis establishes local results under explicit assumptions near an appropriate equilibrium, not unconditional convergence of every large GAN. "}<a href={"https://proceedings.mlr.press/v80/mescheder18a/mescheder18a.pdf"}>{"Mescheder et al., §4"}</a>{""}</Prose>

<PenaltyLocationsFigure />

<Prose>{"Lazy application every "}<InlineMath>{"k"}</InlineMath>{" updates can multiply the regularizer by "}<InlineMath>{"k"}</InlineMath>{" to preserve its expected contribution under that sampling schedule. This does not make the finite optimizer trajectory identical, and optimizer adjustments may be needed. Choose and document the procedure rather than treating a popular interval as a theorem."}</Prose>

<H3>{"A penalty can miss a steep region completely"}</H3>

<Prose>{"Consider "}<InlineMath>{"f(x)=x+4\\operatorname{ReLU}(x-1)"}</InlineMath>{". At sampled points −.5, 0 and .5, the derivative is one and the target-one penalty is zero. At "}<InlineMath>{"x=2"}</InlineMath>{", the derivative is five and the unweighted penalty is sixteen. The global Lipschitz constant is five."}</Prose>

<SpectralPenaltyLab />

<H3>{"Per-example gradients need per-example functions"}</H3>

<Prose>{"The usual code obtains gradients of the sum of batch scores. If each score depends only on its own input, this yields each example's input gradient. Batch-dependent operations can break that interpretation."}</Prose>

<Prose>{"For two scalar inputs, define "}<InlineMath>{"f_1=(x_1-x_2)/2"}</InlineMath>{", "}<InlineMath>{"f_2=(x_2-x_1)/2"}</InlineMath>{". Each score has self-derivative .5, but the gradient of "}<InlineMath>{"f_1+f_2"}</InlineMath>{" is zero. Batch centering has coupled the examples, and summing scores cancels their derivatives. This small Jacobian explains why ordinary training-mode BatchNorm is problematic in the standard WGAN-GP calculation. Per-example LayerNorm avoids that specific batch coupling, but its own sensitivity still depends on its formula, gain and epsilon."}</Prose>

<BatchGradientFigure />

<H2>{"6. Implement the mechanism without losing its derivatives"}</H2>

<Prose>{"For dense layers, PyTorch's parametrization API attaches spectral normalization to the weight. Training-mode weight access updates estimated singular vectors; evaluation mode freezes those iterations. Thus the number of accesses is part of the procedure, not simply the number of optimizer steps. A shared forward for recorded and generated inputs is easy to reason about. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.parametrizations.spectral_norm.html"}>{"PyTorch 2.14 spectral normalization"}</a>{""}</Prose>

<Prose>{"The complete experiment below uses the maintained API instead of a custom wrapper. A custom implementation must correctly handle device/dtype buffers and multiple forwards before backward; mutating cached vectors that autograd still needs can invalidate the graph. Do not replace the standard implementation with a shorter wrapper merely to reduce displayed lines."}</Prose>

<Prose>{"For gradient penalty, retain the returned tensor until "}<InlineMath>{"L_D"}</InlineMath>{" is differentiated. Converting it to a Python number with "}<code>{".item()"}</code>{" is appropriate for logging after detaching, but not for the optimized loss. Use "}<code>{"create_graph=True"}</code>{" for the input derivative, detach generator-produced samples in the critic phase, and flatten all non-batch input dimensions when taking each gradient norm. Interpolation needs one scalar per example broadcast over that example's coordinates."}</Prose>

<Prose>{"There are numerical choices too. A zero matrix has no nonzero singular direction and cannot be divided by its norm; a tiny estimate requires an explicit policy. The small investigation defines the zero matrix's normalized result as zero and reports “direction undefined.” The training program uses ordinary nonzero initialization and checks actual finite outputs. Reduced precision can affect both norm estimation and the higher-order derivative calculation; establish an FP32 reference and verify the intended mixed-precision path before using it. No unmeasured universal overhead percentage is needed to explain that extra derivatives cost work."}</Prose>

<Prose>{"At export, evaluation mode plus "}<code>{"remove_parametrizations(layer, \"weight\", leave_parametrized=True)"}</code>{" retains the current effective weight. Removing the parametrization need not restore the raw unnormalized weight. The companion example verifies identical outputs before and after this operation on a small dense layer. A frozen approximate norm remains approximate; exporting it does not upgrade it into a certificate. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.parametrize.remove_parametrizations.html"}>{"PyTorch removal contract"}</a>{""}</Prose>

<SpectralLibraryLab />

<H2>{"7. A complete experiment on recorded digit measurements"}</H2>

<Prose>{"The question is modest: "}<strong>{"how do three declared critic-regularization procedures behave while learning the distribution of two real ink measurements?"}</strong>{" This is small enough to plot every recorded point and inspect a learned score surface."}</Prose>

<Prose>{"The "}<a href={"/learn-code/spectral-normalization-gradient-penalty/digits-400.csv"}>{"offline CSV"}</a>{" contains 400 real 8×8 optical digit images, selected as the first 40 examples of each digit from the scikit-learn copy of the UCI dataset. Each integer pixel is between zero and sixteen. For each half-image, sum its 32 pixels and divide by "}<InlineMath>{"32\\times16=512"}</InlineMath>{". The two resulting coordinates are average normalized ink on the left and right. Class labels are not used for training or splitting. "}<a href={"https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits"}>{"Dataset source and attribution"}</a>{""}</Prose>

<Prose>{"Different images sometimes give the same measurement pair. Keep these equal profiles in the same role, retaining their frequency. A seeded split of 394 unique profile groups produces "}<strong>{"241 fitting images, 79 development images and 80 assessment images"}</strong>{". No coordinate standardization is learned from assessment data. The "}<a href={"/learn-code/spectral-normalization-gradient-penalty/data-provenance.md"}>{"provenance record"}</a>{" retains every source ID, group, split rule, dependency version and data hash. Writer-level independence is not established by this extract."}</Prose>

<Prose>{"The generator has widths 2→24→24→2, ReLU hidden activations and sigmoid outputs, with 722 parameters. The critic has widths 2→24→24→1, leaky-ReLU slope .2 and an unrestricted scalar output, with 697 parameters. There is no BatchNorm. Compare entrywise clipping at .1, target-one gradient penalty with "}<InlineMath>{"\\lambda=10"}</InlineMath>{", and one-iteration spectral normalization on each dense critic weight. All three use the same Wasserstein-style losses."}</Prose>

<Prose>{"For each method, seeds 11, 29 and 47 give paired raw initializations and data/latent draws. Train for 600 generator updates with three critic updates per generator update, batch size 64 and Adam rate .001, betas "}<InlineMath>{"(0,.9)"}</InlineMath>{". Gradient-penalty interpolation has a separate random stream so it does not alter the common data draws. These choices were fixed before the run. Development measurements are recorded at declared steps; they do not select a checkpoint. All nine final models are retained, including unfavorable outcomes."}</Prose>

<H3>{"Measure a distributional discrepancy independently of critic loss"}</H3>

<Prose>{"Draw the same 256 latent vectors for every final generator. Project generated and recorded pairs onto 64 equally spaced unit directions with angles "}<InlineMath>{"j\\pi/64"}</InlineMath>{". In each one-dimensional projection, compute the empirical Wasserstein-1 distance, then average. Sorting and integrating the empirical cumulative-distribution difference gives each one-dimensional value, including when sample counts differ."}</Prose>

<Prose>{"This is a "}<strong>{"finite directional average of empirical W1"}</strong>{", not exact two-dimensional W1, FID or a log likelihood. Its units are normalized ink coordinates. It can miss differences between the chosen projections and is subject to finite-sample variation. A simple baseline samples 256 fitting profiles with replacement using a fixed seed; it needs no neural training. Its assessment discrepancy is "}<strong>{".010039"}</strong>{"."}</Prose>

<NeuralTable caption={"Measure a distributional discrepancy independently of critic loss"} headers={[<>{"Procedure"}</>,<>{"Seed 11"}</>,<>{"Seed 29"}</>,<>{"Seed 47"}</>]} rows={[[<>{"Entrywise clipping"}</>,<>{".047078"}</>,<>{".031269"}</>,<>{".035174"}</>],[<>{"Gradient penalty"}</>,<>{".150081"}</>,<>{".277581"}</>,<>{".318414"}</>],[<>{"Spectral normalization"}</>,<>{".018989"}</>,<>{".013203"}</>,<>{".016699"}</>]]} />

<Prose>{"Lower means closer under this metric. Spectral normalization is best among these nine neural runs, but the empirical-resampling baseline is better still. The experiment therefore does not show that a neural generator is necessary for this task. Nor does one fixed budget settle which regularizer is best after suitable tuning on a different problem."}</Prose>

<Prose>{"The critic measurements answer another question:"}</Prose>

<NeuralTable caption={"Measure a distributional discrepancy independently of critic loss"} headers={[<>{"Seed 11 critic"}</>,<>{"Maximum gradient on 80 assessment points"}</>,<>{"Maximum on a 41×41 input grid"}</>,<>{"Product of exact effective matrix norms"}</>]} rows={[[<>{"Clipping"}</>,<>{".008889"}</>,<>{".018682"}</>,<>{".193269"}</>],[<>{"Gradient penalty"}</>,<>{"1.006960"}</>,<>{"1.088335"}</>,<>{"3.471948"}</>],[<>{"Spectral normalization"}</>,<>{".007496"}</>,<>{".237637"}</>,<>{"1.000043"}</>]]} />

<Prose>{"The spectral product is slightly above one because training used an estimate. Its much smaller observed gradients illustrate a loose product bound. The gradient-penalty model's individual assessment gradient norms range from .737725 to 1.006960: their maximum is near the target, while its generated profiles remain poor. Better local sensitivity behavior is not equivalent to better generated data."}</Prose>

<SpectralMeasuredLab />

<Prose>{"For a null that tests the function rather than its label, swap the two latent coordinates and simultaneously swap the two columns of the first generator weight matrix. Every generated output remains unchanged. Swapping only the input coordinates generally changes the output. The saved models verify this distinction; it is a change of coordinate naming versus a change of input to a fixed function."}</Prose>

<H3>{"Run the declared experiment"}</H3>

<Prose>{"Save the CSV beside the program. The author run used Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1 and PyTorch 2.14.0+cpu with one CPU thread. Install compatible packages in an isolated environment, then run "}<code>{"python critic-regularization-study.py"}</code>{". The program reads the real extract, creates the roles, fits all nine models, measures them, and writes the weights and results. No dataset download or hidden training loop is required."}</Prose>

<Prose>{"The "}<a href={"/learn-code/spectral-normalization-gradient-penalty/critic-regularization-study.py"}>{"complete program"}</a>{" is printed below. It intentionally uses small explicit training steps and saves the evidence needed to understand the results. The "}<a href={"/learn-code/spectral-normalization-gradient-penalty/sensitivity-calculations.py"}>{"separate sensitivity calculations"}</a>{" reproduce the exact examples and check frozen-model inference without fitting again."}</Prose>

<SpectralProgram />

<H3>{"What you can now implement and deliberately change"}</H3>

<Prose>{"The two methods have different source owners. In "}<a href={"/learn-code/spectral-normalization-gradient-penalty/sensitivity-calculations.py"}>{"sensitivity-calculations.py"}</a>{", "}<code>{"power_trace"}</code>{" constructs repeated matrix/vector products and normalization, and the normalization-gradient calculation differentiates through the weight-dependent scale. In "}<a href={"/learn-code/spectral-normalization-gradient-penalty/critic-regularization-study.py"}>{"critic-regularization-study.py"}</a>{", "}<code>{"gradient_penalty"}</code>{" constructs interpolation points, input derivatives and the differentiable norm penalty; "}<code>{"main"}</code>{" composes that operation with a complete critic/generator training loop. Nothing in the penalty is delegated to an unexplained “WGAN loss” call."}</Prose>

<Prose>{"The ordinary spectral-normalization route is the same program's "}<code>{"torch.nn.utils.parametrizations.spectral_norm(layer, n_power_iterations=1)"}</code>{". Its trainable original weight, computed normalized weight and power-vector buffers are distinct state. Calling a layer twice in training mode can perform two power-vector updates; the joined real/fake forward intentionally gives both groups one common effective weight. Critic evaluation mode freezes power-vector iteration during the generator step, while "}<code>{"requires_grad_(False)"}</code>{" freezes critic parameters: input derivatives still connect the generator to its objective. The export calculation in "}<code>{"sensitivity-calculations.py"}</code>{" checks that removing the parametrization with "}<code>{"leave_parametrized=True"}</code>{" preserves the current function. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.parametrizations.spectral_norm.html"}>{"Maintained API and mode semantics"}</a>{"."}</Prose>

<Prose>{"This is the useful division of control: own the desired penalty, sample distribution and update schedule; delegate module registration, persistent buffers and serialization to the maintained parametrization. One power iteration is an estimator, not an exact largest singular value. Comparing it with an SVD on a tiny matrix diagnoses estimation error; it does not make the SVD algorithm the usual per-training-step choice. Dense power iteration costs O(mn) per iteration with O(m+n) vector state beyond the weight; an exact SVD has a different cost. Input-gradient penalties retain an extra derivative graph and require per-example critic independence for the sum trick used here."}</Prose>

<Prose>{""}<strong>{"Change the contract."}</strong>{" Replace the interpolated unit-target penalty by an R1 penalty on real inputs, keeping the critic objective's other terms fixed. Implement the new quantity rather than changing only its caption."}</Prose>

<details><summary>Hint and reasoned solution</summary>

<Prose>{"Use "}<code>{"points = real.detach().requires_grad_(True)"}</code>{", evaluate the critic, and obtain "}<code>{"gradients = torch.autograd.grad(values.sum(), points, create_graph=True)[0]"}</code>{". With a declared coefficient λ, the zero-centered term is "}<code>{"λ * gradients.flatten(1).square().sum(1).mean()"}</code>{". There is no subtraction of one and no fake/interpolation sampling. Some conventions write γ/2 instead of λ; map the coefficient explicitly. Preserve "}<code>{"create_graph=True"}</code>{" when updating critic parameters through this derivative. For the independent linear oracle D(x)=wᵀx, the added objective is λ‖w‖² and its weight derivative is 2λw. This is different from the existing unit-target derivative and remains meaningful when the input locations change. It is not a global Lipschitz certificate."}</Prose>

</details>

<H2>{"8. Useful connections beyond this GAN"}</H2>

<H3>{"A robustness margin needs the right output difference"}</H3>

<Prose>{"A classifier chooses the largest logit. Suppose its two logits at "}<InlineMath>{"x=(2,0)"}</InlineMath>{" are produced by "}<InlineMath>{"F(x)=(x_1,x_2)"}</InlineMath>{". The winning gap is two. The vector-valued map has Euclidean Lipschitz constant one, but the "}<strong>{"difference"}</strong>{" "}<InlineMath>{"F_1-F_2"}</InlineMath>{" has constant "}<InlineMath>{"\\sqrt2"}</InlineMath>{". Any perturbation of norm strictly below "}<InlineMath>{"2/\\sqrt2=\\sqrt2"}</InlineMath>{" preserves the positive gap. Perturbation "}<InlineMath>{"(-1,1)"}</InlineMath>{", whose norm is exactly "}<InlineMath>{"\\sqrt2"}</InlineMath>{", reaches a tie."}</Prose>

<Prose>{"More generally, if a joint logit vector has bound "}<InlineMath>{"L"}</InlineMath>{", a pairwise logit difference has bound at most "}<InlineMath>{"\\sqrt2L"}</InlineMath>{". A positive gap "}<InlineMath>{"m"}</InlineMath>{" therefore gives the sufficient radius "}<InlineMath>{"m/(\\sqrt2L)"}</InlineMath>{" for that competitor. If each logit separately has bound "}<InlineMath>{"L"}</InlineMath>{", the direct sum bound is "}<InlineMath>{"2L"}</InlineMath>{". For multiple competitors, take the smallest valid radius. Include input preprocessing and the domain in the bound. A power-iteration estimate or a sampled gradient maximum alone is insufficient evidence for this certificate."}</Prose>

<Prose>{"This is a useful application of the same geometry: the matrix stretch becomes a bound on a decision change. It is not a claim that adding spectral normalization alone makes a classifier robust to every perturbation."}</Prose>

<SpectralMarginLab />

<H3>{"Bounded slope can still limit expressive power"}</H3>

<Prose>{"ReLU can remove a component's derivative entirely on its inactive side. A network constrained at every layer may lose useful gradient magnitude through repeated such operations. "}<strong>{"GroupSort"}</strong>{" instead sorts small groups of activations; within a region where their order is fixed, it permutes components and preserves their Euclidean length. Sorting supplies nonlinearity without discarding that local derivative magnitude."}</Prose>

<Prose>{"For two values "}<InlineMath>{"(a,b)"}</InlineMath>{", GroupSort returns "}<InlineMath>{"(\\min(a,b),\\max(a,b))"}</InlineMath>{". Their order can change as inputs change, producing a nonlinear piecewise-defined function. This helps explain why the activation and the norm constraint must be designed together. The original universal-approximation theorem uses specified mixed norms: a first-layer "}<InlineMath>{"p\\to\\infty"}</InlineMath>{" bound and subsequent infinity-norm bounds. It does not establish that any spectrally normalized ReLU network, or every Euclidean GroupSort construction, approximates every Lipschitz function. "}<a href={"https://proceedings.mlr.press/v97/anil19a/anil19a.pdf"}>{"Anil et al., architecture and Theorem 3"}</a>{""}</Prose>

<SpectralGroupSortFigure />

<H3>{"Continuous-time sensitivity is about trajectories too"}</H3>

<Prose>{"In an ODE "}<InlineMath>{"dh/dt=f(h,t)"}</InlineMath>{", a Lipschitz bound in "}<InlineMath>{"h"}</InlineMath>{", together with appropriate continuity conditions, helps establish uniqueness and bound how two trajectories separate. A typical bound is "}<InlineMath>{"\\|h(t)-\\widetilde h(t)\\|\\le e^{Lt}\\|h(0)-\\widetilde h(0)\\|"}</InlineMath>{". An upper bound on growth is not a promise of contraction. The simple field "}<InlineMath>{"f(h)=Lh"}</InlineMath>{" has bounded slope but unbounded values as "}<InlineMath>{"|h|"}</InlineMath>{" grows and exponentially separating trajectories when "}<InlineMath>{"L>0"}</InlineMath>{"."}</Prose>

<Prose>{"The later "}<a href={"/learn/path/full-curriculum/neural-ode-continuous-depth-models?module=deep-learning-fundamentals"}>{"Neural ODE & Continuous-Depth Models"}</a>{" lesson develops that connection and the separate numerical-solver questions. A layer norm bound alone does not prescribe a safe solver step size."}</Prose>

<SpectralOdeFigure />

<H3>{"Diagnose the observed failure before changing methods"}</H3>

<Prose>{"If a critic has non-finite values, first inspect input scales, actual norms and derivative paths. If a penalty stays near its target but samples remain poor, inspect the generated distribution, capacity, update balance and training trajectory. If a tight product bound removes too much sensitivity, consider whether the architecture and desired constraint fit the task. Combining spectral normalization and a sampled penalty can be meaningful because they act through different mechanisms."}</Prose>

<Prose>{"Compare equal objectives and report changed choices. Critic-loss signs and offsets, penalty terms and output scales make raw losses across different setups difficult to compare. For a binary discriminator, a summed real/fake BCE near "}<InlineMath>{"2\\log2"}</InlineMath>{" is the value obtained by .5 predictions, not proof of perfect separation. A near-zero Wasserstein-style critic difference can indicate indistinguishable distributions or an uninformative critic. Use independent data-space or task-appropriate evaluation to tell these possibilities apart."}</Prose>

<section className="lesson-ending lesson-ending--practice" data-lesson-ending="practice"><H2>{"9. Practice and transfer"}</H2>

<Prose>{"Attempt each question before opening its help. The early questions check mechanisms; later ones ask you to diagnose a system."}</Prose>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"1. A changed matrix"}</H3>

<Prose>{"For "}<InlineMath>{"W=\\operatorname{diag}(4,2)"}</InlineMath>{", find its spectral norm, Frobenius norm and exact unit-normalized singular values. Then normalize "}<InlineMath>{".2I"}</InlineMath>{": does it shrink?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"For a diagonal matrix with nonnegative entries, the entries are its singular values. Exact unit normalization divides by the largest."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The norms are 4 and √20; normalized singular values are 1 and .5. The matrix .2I becomes I, so it grows. A cap-only operation leaves .2I unchanged. Normalization to a boundary and projection into a bounded set are different operations."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"2. A direction the estimator cannot see"}</H3>

<Prose>{"For "}<InlineMath>{"W=\\operatorname{diag}(2,5)"}</InlineMath>{", start power iteration at "}<InlineMath>{"u=(1,0)"}</InlineMath>{". What estimate persists? What is the true norm after dividing by it? Would adding a small second component change the long-run behavior?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Track which coordinates matrix multiplication can create from a zero coordinate."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The estimate stays 2 and the resulting true norm is 2.5. In exact arithmetic the leading direction has no component to amplify. A nonzero second component allows repeated multiplication to amplify that direction relative to the first, eventually approaching estimate 5. The number of steps depends on the initial component and spectral gap."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"3. A valid bound through a residual path"}</H3>

<Prose>{"Two consecutive linear layers have bounds .8 and .6, with ReLU between them. They form a residual branch added to the input. Give a valid block bound. Does a learned multiplier of 2 outside the block preserve it?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"First compose the branch, then account for addition, then the multiplier."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The branch bound is .48; the residual block bound is 1.48; the scaled block bound is 2.96. These are upper bounds, not assertions that some input pair must attain them. A claim of .48 for the whole residual block omitted the identity path."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"4. A penalty that sees the wrong region"}</H3>

<Prose>{"Let "}<InlineMath>{"f(x)=x+2\\operatorname{ReLU}(x-2)"}</InlineMath>{". Sample only 0, 1 and 1.5. What is the unweighted target-one gradient penalty? What changes if the last probe moves to 3? What is the global Lipschitz constant?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"The extra slope begins only after the kink; average the three squared deviations."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The first penalty is 0. With probes 0, 1, 3 the slopes are 1, 1, 3, so the mean penalty is 4/3. The global bound is 3. Keeping all probes below 2 leaves the violation unobserved; increasing the penalty coefficient cannot penalize a region that this sample never measures."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"5. Repair the optimization graph"}</H3>

<Prose>{"A training loop calculates "}<code>{"penalty = gradient_penalty(...).item()"}</code>{", adds it to the critic loss, and detaches "}<code>{"critic(generator(z))"}</code>{" during the generator update. Explain both failures and their repairs."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Distinguish a logged number from a differentiable tensor, and frozen parameters from frozen inputs."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The Python scalar carries no parameter derivative, so that penalty cannot regularize the critic. Keep its tensor in the loss; detach only a separate value for logging. Detaching the generator's scored output removes its learning path. Freeze critic parameters while preserving its derivative with respect to the generated input. Detach generated samples only when they serve as fixed inputs for the critic phase."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"6. Interpret the real experiment"}</H3>

<Prose>{"The seed 11 gradient-penalty critic has assessment gradient maximum about 1.007, yet generated-profile discrepancy is .150081. The empirical-resampling baseline gives .010039. What conclusion is supported, and what comparison is still missing?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"The three numbers measure two different questions; one setting per procedure does not isolate its best possible performance."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The largest measured gradient is near the requested target, but the resulting generator fits these profiles poorly under the declared directional metric. The simple baseline is better in this experiment. That does not refute gradient penalty in general or prove spectral normalization universally superior. A broader comparison would predeclare a development-based tuning budget, keep assessment separate, compare several seeds and evaluate the actual intended data representation. Producing realistic digit images is a different task from producing two ink averages."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"7. Convolution changes when windows overlap"}</H3>

<Prose>{"For the four-input stride-two kernel "}<InlineMath>{"[2,2]"}</InlineMath>{", calculate the full operator norm and the norm after normalizing the stored kernel. Why is the answer different for a valid stride-one application to three inputs?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Write the full matrices. Scaling a matrix by 2 scales every singular value by 2."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The disjoint operator is [[2,2,0,0],[0,0,2,2]], with norm 2√2. The stored kernel has that same norm, so normalization gives full norm 1. The overlapping three-input operator has norm 2√3; normalization by 2√2 leaves √(3/2). Weight sharing and overlap, not the number of stored coefficients alone, determine the full map."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"8. A robustness calculation with units"}</H3>

<Prose>{"A two-logit network has a verified joint Euclidean Lipschitz upper bound 2 on normalized inputs and a winning logit gap .8. Give a sufficient perturbation radius in normalized-input units. What else is needed before stating a radius in raw sensor units?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Use a bound for the difference of two coordinates, then compose preprocessing."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The sufficient strict radius is .8/(2√2)≈.282843. Raw-unit interpretation needs the normalization map and its operator bound, along with the domain on which the network bound applies. If coordinate scaling is anisotropic, one scalar conversion may be overly conservative; state the input norm and transform explicitly. The bound must be verified, not merely a sampled gradient maximum."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"9. Plan a useful follow-up"}</H3>

<Prose>{"Your generated profiles form a narrow curve while the recorded profiles occupy a broader region. Propose a next experiment that distinguishes limited generator capacity, insufficient training and an evaluation artifact without choosing settings using assessment results."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Separate interventions and keep the observation unit, roles and metric definitions fixed."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Keep the existing split and select a bounded development-only comparison: first longer training at fixed architecture, then a changed generator width or latent dimension with an explicit compute budget. Use paired seeds where possible and retain all outcomes. Inspect actual point clouds and add a predeclared complementary discrepancy or coverage measure; do not rename critic loss as quality. Choose using development evidence, then evaluate the selected procedure on untouched assessment data. Existing assessment results have already been inspected, so a strong new confirmatory claim would need fresh assessment data rather than pretending this set is unseen again."}</Prose>

</details>

<Prose>{""}<strong>{"Ready to continue:"}</strong>{" you can trace critic versus generator derivatives, distinguish exact norms from estimates and sampled measurements, calculate one normalization and one penalty update, and explain an unfavorable result without changing the question after seeing it. The next topic in the module is "}<a href={"/learn/path/full-curriculum/modern-hopfield-networks?module=deep-learning-fundamentals"}>{"Modern Hopfield Networks"}</a>{", which returns to energy-based retrieval and connects it to attention. It remains the next lesson even if publication timing differs."}</Prose></div></section>

<section className="lesson-ending lesson-ending--resources" data-lesson-ending="resources"><H2>{"References and another way to learn"}</H2>

<ul><li>{""}<a href={"https://arxiv.org/pdf/1802.05957"}>{"Spectral Normalization for Generative Adversarial Networks"}</a>{", Miyato et al. Start with §2 and Appendix A for the method; Appendix F develops its derivative. Original 2018 experiments describe their own settings, not current hardware rankings."}</li><li>{""}<a href={"https://proceedings.neurips.cc/paper_files/paper/2017/file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf"}>{"Improved Training of Wasserstein GANs"}</a>{", Gulrajani et al. Read the sampling rule, Algorithm 1 and no-BatchNorm explanation in §4 after working the penalty example."}</li><li>{""}<a href={"https://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf"}>{"Wasserstein GAN"}</a>{", Arjovsky et al. The point-mass example and §2–3 connect distance, continuity, critic constraints and the generator sign."}</li><li>{""}<a href={"https://proceedings.mlr.press/v80/mescheder18a/mescheder18a.pdf"}>{"Which Training Methods for GANs Do Actually Converge?"}</a>{", Mescheder et al. Advanced reading for zero-centered penalties and the assumptions behind local convergence results."}</li><li>{""}<a href={"https://arxiv.org/pdf/1805.10408"}>{"The Singular Values of Convolutional Layers"}</a>{", Sedghi et al. Follow the full-operator viewpoint and check circular-boundary assumptions before reusing a formula."}</li><li>{""}<a href={"https://proceedings.mlr.press/v97/anil19a/anil19a.pdf"}>{"Sorting Out Lipschitz Function Approximation"}</a>{", Anil et al. Explains gradient-norm preservation and a precisely stated approximation theorem; compare its norm choices with the Euclidean examples here."}</li><li>{""}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.parametrizations.spectral_norm.html"}>{"PyTorch spectral-normalization API"}</a>{" and "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.parametrize.remove_parametrizations.html"}>{"parametrization removal"}</a>{". Versioned implementation references for the executed code; inspect train/eval behavior when changing versions."}</li><li>{""}<a href={"https://deepgenerativemodels.github.io/notes/gan/"}>{"Stanford CS236 GAN notes"}</a>{". A shorter alternate introduction to the generator/discriminator game and sample-based evaluation. Its broad introductory simplifications should be read alongside the explicit assumptions in this lesson."}</li><li>{""}<a href={"https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans"}>{"Build Basic GANs, DeepLearning.AI"}</a>{". An alternate video-and-exercise route with introductory GANs and Wasserstein/gradient-penalty material; suitable after basic PyTorch. The course and its listed curriculum were checked, not all videos watched. Access to graded material may require enrollment; this lesson is self-contained. The course is also linked from "}<a href={"https://cs236g.stanford.edu/"}>{"Stanford CS236G's schedule"}</a>{"."}</li><li>{""}<a href={"/learn-code/spectral-normalization-gradient-penalty/data-provenance.md"}>{"Offline study and provenance"}</a>{". Download the real extract and complete programs to reproduce this lesson's own measurements; these are not published benchmark results."}</li></ul></section>
</div>};
