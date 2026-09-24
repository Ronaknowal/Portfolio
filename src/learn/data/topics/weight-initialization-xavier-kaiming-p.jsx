// Complete conserved prepared manuscript, statically rendered at authoring time.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { InitializationSignalLab, InitializationMomentsLab, InitializationGeometryLab, InitializationSpectrumFigure, InitializationSymmetryLab, InitializationTrainingLab, InitializationWidthLab, InitializationPrecisionFigure, InitializationProgram, initializationAsset } from '../../components/lesson-labs/WeightInitializationLabs.jsx';
export default {
 title: 'Weight Initialization: Xavier, Kaiming, Orthogonal Methods & μP',
 readTime: '~65 min read + experiments and practice; optional μP route ~30 min',
 hasIntegratedGuide: true,
 content: () => <div className="neural-lesson initialization-lesson"><LessonIntro prerequisites="Weighted sums, activations and backpropagation. Statistical moments and singular directions are refreshed locally; the prior loss, normalization and transfer lessons own their full mechanisms." sections={[["a-signal-can-disappear-before-learning-starts","A signal can disappear before learning starts"],["what-the-scale-calculation-actually-preserves","What the scale calculation actually preserves"],["choose-a-recipe-for-the-operation-it-initializes","Choose a recipe for the operation it initializes"],["average-preservation-can-hide-a-collapsed-direction","Average preservation can hide a collapsed direction"],["construct-an-orthogonal-draw-then-match-a-width-aware-optimizer","Construct an orthogonal draw, then match a width-aware optimizer"],["why-equal-hidden-units-can-stay-equal","Why equal hidden units can stay equal"],["run-a-complete-initialization-comparison-on-real-inputs","Run a complete initialization comparison on real inputs"],["p-changing-width-changes-more-than-parameter-count","μP: changing width changes more than parameter count"],["deeper-tools-and-practical-failure-checks","Deeper tools and practical failure checks"],["practice-use-the-mechanism-on-a-changed-case","Practice: use the mechanism on a changed case"],["another-way-to-learn-and-what-comes-next","Another way to learn, and what comes next"]]}>Choose a starting state by the signals, directions and updates it preserves. Follow the core route first; μP adds a separate width-scaling route.</LessonIntro>
<Prose>{""}<strong>{"Explore as you read."}</strong>{" Inspect saved initialization/seed traces; edit four activation values, singular directions, depth and width/rate scaling. Show forward/backward second moments, means/variance, directional gain and shape/update formulas together. Continuous tiny models are distinct from selectors over measured training records. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose an initialization/parameterization by the signal and update behavior it preserves, without treating average scale as every-direction stability."}</Prose>

<Prose>{"A network begins making predictions before it has learned anything. Its initial weights determine whether useful differences between inputs survive the journey through its layers—and whether a change in an early weight can still affect the loss."}</Prose>

<Prose>{"Think of passing a sound through twenty amplifiers. A small gain error repeated twenty times can make the signal nearly inaudible or enormously loud. A neural network adds another complication: nonlinear gates can remove parts of the signal. Choosing a starting scale is therefore a problem about a whole sequence of transformations, not simply drawing “small random numbers.”"}</Prose>

<Prose>{""}<strong>{"First pass:"}</strong>{" follow the signal example, the second-moment calculation, the initialization recipes, the geometry counterexample, and the complete digit experiment. Then solve the first three practice problems. The μP section is a second route for learning how to change network width while keeping a meaningful training procedure. LSUV, Fixup, and precision details are reference branches; they are not prerequisites for the next lesson."}</Prose>

<Prose>{"You need weighted sums, a nonlinear activation, and the idea that backpropagation multiplies local sensitivities. We will refresh the statistics and matrix geometry where they are used. The "}<a href={"/learn/path/full-curriculum/transfer-learning-fine-tuning-strategies?module=deep-learning-fundamentals"}>{"previous lesson on transfer learning"}</a>{" reused learned weights. Here we study fresh initialization. Later, "}<strong>{"μTransfer"}</strong>{" will mean transferring selected hyperparameters across widths, which is a different operation."}</Prose>

<H2>{"A signal can disappear before learning starts"}</H2>

<Prose>{"Suppose every hidden layer has 64 inputs, zero bias, and a ReLU, which replaces negative values with zero. Consider three normal distributions for its weights:"}</Prose>

<NeuralTable caption={"A signal can disappear before learning starts"} headers={[<>{"Weight standard deviation"}</>,<>{"Intended question"}</>]} rows={[[<>{"0.01"}</>,<>{"Are small values automatically safe?"}</>],[<>{""}<InlineMath>{"\\sqrt{2/64}\\approx0.177"}</InlineMath>{""}</>,<>{"Does accounting for the number of inputs and the ReLU help?"}</>],[<>{"0.5"}</>,<>{"What happens when repeated layers amplify too much?"}</>]]} />

<Prose>{"The experiment accompanying this lesson sends the same 128 Gaussian input vectors through twenty layers. These are diagnostic inputs, not digit images. It measures"}</Prose>

<div className="neural-equation"><MathBlock>{"q=\\operatorname{mean}(h^2),"}</MathBlock></div>

<Prose>{"the mean squared activation, pooling every row and coordinate of that layer. A useful signal does not have to maintain exactly the same "}<InlineMath>{"q"}</InlineMath>{". But changes by dozens of orders of magnitude deserve investigation."}</Prose>

<Prose>{"Actual float64 results for seed 1:"}</Prose>

<NeuralTable caption={"A signal can disappear before learning starts"} headers={[<>{"Initialization"}</>,<>{"Layer 1 "}<InlineMath>{"q"}</InlineMath>{""}</>,<>{"Layer 5 "}<InlineMath>{"q"}</InlineMath>{""}</>,<>{"Layer 10 "}<InlineMath>{"q"}</InlineMath>{""}</>,<>{"Layer 20 "}<InlineMath>{"q"}</InlineMath>{""}</>]} rows={[[<>{"Normal, std 0.01"}</>,<>{"0.00294"}</>,<>{""}<InlineMath>{"3.61\\times10^{-13}"}</InlineMath>{""}</>,<>{""}<InlineMath>{"1.36\\times10^{-25}"}</InlineMath>{""}</>,<>{""}<InlineMath>{"9.06\\times10^{-51}"}</InlineMath>{""}</>],[<>{"Xavier, gain 1"}</>,<>{"0.459"}</>,<>{"0.0337"}</>,<>{"0.00118"}</>,<>{""}<InlineMath>{"6.81\\times10^{-7}"}</InlineMath>{""}</>],[<>{"Kaiming, ReLU"}</>,<>{"0.918"}</>,<>{"1.077"}</>,<>{"1.207"}</>,<>{"0.714"}</>],[<>{"Orthogonal, gain "}<InlineMath>{"\\sqrt2"}</InlineMath>{""}</>,<>{"0.976"}</>,<>{"0.682"}</>,<>{"0.630"}</>,<>{"0.232"}</>],[<>{"Normal, std 0.5"}</>,<>{"7.343"}</>,<>{"35,302"}</>,<>{""}<InlineMath>{"1.30\\times10^9"}</InlineMath>{""}</>,<>{""}<InlineMath>{"8.24\\times10^{17}"}</InlineMath>{""}</>]]} />

<Prose>{"The input "}<InlineMath>{"q"}</InlineMath>{" was 0.970. These are observations from one finite network, not theoretical curves. The downloadable record also contains seeds 2 and 3. In particular, the orthogonal run drifted downward; its name does not guarantee a flat line after nonlinearities."}</Prose>

<InitializationSignalLab />

<Prose>{"For the backward probe, the program forms a scalar by multiplying the final activations by a fixed random array and summing. It then differentiates that scalar with respect to every layer's activations. The input gradient RMS is "}<InlineMath>{"9.54\\times10^{-26}"}</InlineMath>{" for small initialization, 0.848 for Kaiming, and "}<InlineMath>{"9.10\\times10^8"}</InlineMath>{" for large initialization. This measures sensitivity to one chosen output direction. It does not measure all possible directions or prove that a classifier will train."}</Prose>

<H2>{"What the scale calculation actually preserves"}</H2>

<Prose>{"A "}<strong>{"mean"}</strong>{" describes location. A "}<strong>{"variance"}</strong>{" measures squared spread around that mean. A "}<strong>{"second moment"}</strong>{" measures squared distance from zero:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\operatorname{Var}(z)=E[z^2]-(E[z])^2."}</MathBlock></div>

<Prose>{"They coincide when the mean is zero. After ReLU, that is usually no longer true."}</Prose>

<Prose>{"Take four equally weighted values:"}</Prose>

<NeuralTable caption={"What the scale calculation actually preserves"} headers={[<>{""}</>,<>{"Values"}</>,<>{"Mean"}</>,<>{"Second moment"}</>,<>{"Variance"}</>]} rows={[[<>{"Before ReLU"}</>,<>{"−2, −1, 1, 2"}</>,<>{"0"}</>,<>{"2.5"}</>,<>{"2.5"}</>],[<>{"After ReLU"}</>,<>{"0, 0, 1, 2"}</>,<>{"0.75"}</>,<>{"1.25"}</>,<>{"0.6875"}</>]]} />

<Prose>{"The second moment halved. The variance did "}<strong>{"not"}</strong>{" halve: the output also moved away from zero. Follow the four values as the negative ones move to zero, then compare their distances from zero with their distances from the new mean."}</Prose>

<InitializationMomentsLab />

<Prose>{"Now consider one preactivation,"}</Prose>

<div className="neural-equation"><MathBlock>{"z=\\sum_{i=1}^{n}w_i x_i."}</MathBlock></div>

<Prose>{"Here "}<InlineMath>{"n"}</InlineMath>{" is "}<strong>{"fan-in"}</strong>{", the number of contributions to this output. Assume initially independent, zero-mean weights with variance "}<InlineMath>{"s^2"}</InlineMath>{", independent of the input vector. For a fixed input, cross terms vanish when averaging over those random weights:"}</Prose>

<div className="neural-equation"><MathBlock>{"E_w[z^2\\mid x]=s^2\\sum_i x_i^2."}</MathBlock></div>

<Prose>{"If the input coordinates share second moment "}<InlineMath>{"q"}</InlineMath>{", averaging over inputs gives "}<InlineMath>{"E[z^2]=ns^2q"}</InlineMath>{". Input coordinates do not have to have zero mean for this calculation. What removes the cross terms is the assumption about the weights. After training, weights depend on the data and such independence is no longer a reliable description."}</Prose>

<Prose>{"For a symmetric preactivation distribution, exactly half its squared mass is on either side of zero:"}</Prose>

<div className="neural-equation"><MathBlock>{"E[\\operatorname{ReLU}(z)^2]=\\tfrac12 E[z^2]."}</MathBlock></div>

<Prose>{"Combining these equations gives"}</Prose>

<div className="neural-equation"><MathBlock>{"q_{\\mathrm{next}}=\\tfrac12 ns^2q."}</MathBlock></div>

<Prose>{"Choosing "}<InlineMath>{"s^2=2/n"}</InlineMath>{" makes the approximate layer-to-layer factor one. This is the ReLU "}<strong>{"Kaiming"}</strong>{", or "}<strong>{"He"}</strong>{", rule. The symmetry condition matters; a substantial bias can change how much of the distribution is removed. The original rectifier analysis also develops the backward calculation, with assumptions about gates and incoming derivatives. "}<a href={"https://arxiv.org/pdf/1502.01852"}>{"He et al., §2.2"}</a>{""}</Prose>

<Prose>{"There are several different averages here. The derivation averages over random initializations and input assumptions. The program measures one realized tensor. A mean of individual coordinates' variances across examples would be yet another statistic. The chart reports pooled second moment explicitly, so it does not disguise those differences."}</Prose>

<H2>{"Choose a recipe for the operation it initializes"}</H2>

<Prose>{"For a linear layer stored as an "}<InlineMath>{"m\\times n"}</InlineMath>{" matrix, fan-in is "}<InlineMath>{"n"}</InlineMath>{" and fan-out is "}<InlineMath>{"m"}</InlineMath>{". Forward signals collect "}<InlineMath>{"n"}</InlineMath>{" terms. Backward signals collect "}<InlineMath>{"m"}</InlineMath>{" terms."}</Prose>

<NeuralTable caption={"Choose a recipe for the operation it initializes"} headers={[<>{"Method"}</>,<>{"Listed variance or construction"}</>,<>{"Starting use"}</>]} rows={[[<>{"LeCun normal"}</>,<>{""}<InlineMath>{"1/n"}</InlineMath>{""}</>,<>{"Preserve a linear forward second moment under the assumptions above; also part of particular self-normalizing recipes"}</>],[<>{"Xavier/Glorot normal"}</>,<>{""}<InlineMath>{"2/(n+m)"}</InlineMath>{""}</>,<>{"Balance forward and backward scale for approximately linear, centered activations"}</>],[<>{"Kaiming normal, fan-in"}</>,<>{""}<InlineMath>{"2/n"}</InlineMath>{" for ReLU"}</>,<>{"Preserve the forward second-moment scale through ReLU"}</>],[<>{"Kaiming normal, fan-out"}</>,<>{""}<InlineMath>{"2/m"}</InlineMath>{" for ReLU"}</>,<>{"Prioritize the corresponding backward scale"}</>],[<>{"Orthogonal"}</>,<>{"Construct orthogonal rows or columns, then multiply by a gain"}</>,<>{"Control the linear map's geometry, with attention to dimensions and later nonlinearities"}</>]]} />

<Prose>{"For a normal draw with variance "}<InlineMath>{"v"}</InlineMath>{", the standard deviation is "}<InlineMath>{"\\sqrt v"}</InlineMath>{". For a uniform draw on "}<InlineMath>{"[-a,a]"}</InlineMath>{", the variance is "}<InlineMath>{"a^2/3"}</InlineMath>{", so use "}<InlineMath>{"a=\\sqrt{3v}"}</InlineMath>{". A gain "}<InlineMath>{"g"}</InlineMath>{" multiplies the weights and therefore multiplies variance by "}<InlineMath>{"g^2"}</InlineMath>{". The listed Kaiming variances already include the ReLU gain; do not apply it twice."}</Prose>

<Prose>{"For example, a 100-input, 25-output linear layer has Xavier variance "}<InlineMath>{"2/125=0.016"}</InlineMath>{". Its normal standard deviation is about 0.1265 and its uniform bound is about 0.2191. The idealized forward factor is "}<InlineMath>{"100(0.016)=1.6"}</InlineMath>{", while the backward factor is "}<InlineMath>{"25(0.016)=0.4"}</InlineMath>{". Xavier compromises; it cannot preserve both exactly when dimensions differ. In extreme aspect ratios one of those factors can be arbitrarily small."}</Prose>

<Prose>{"For leaky ReLU with negative slope "}<InlineMath>{"a"}</InlineMath>{", the squared multiplier under symmetry is "}<InlineMath>{"(1+a^2)/2"}</InlineMath>{". This gives variance "}<InlineMath>{"2/[(1+a^2)n]"}</InlineMath>{". Setting "}<InlineMath>{"a=0"}</InlineMath>{" recovers ReLU. For tanh, a variance calculation near zero is only a local approximation because tanh saturates. A library's suggested tanh gain is a starting convention, not proof of constant moments at all depths. GELU and SiLU similarly deserve measurement in the actual architecture."}</Prose>

<Prose>{"Xavier's original study connects activation saturation, initialization, and observed training behavior. Its assumptions and empirical comparisons motivate a diagnostic approach rather than a universal promise. "}<a href={"https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"}>{"Glorot and Bengio, §§3–5"}</a>{""}</Prose>

<Prose>{"Here is a complete small API example:"}</Prose>

<CodeBlock language={"python"}>{"import torch\nfrom torch import nn\n\ntorch.manual_seed(7)\nhidden = nn.Linear(100, 25)\nhead = nn.Linear(25, 3)\nnn.init.kaiming_normal_(hidden.weight, mode=\"fan_in\", nonlinearity=\"relu\")\nnn.init.zeros_(hidden.bias)\nnn.init.xavier_normal_(head.weight)\nnn.init.zeros_(head.bias)\nx = torch.ones(2, 100)\nlogits = head(torch.relu(hidden(x)))\nprint(logits.shape)  # torch.Size([2, 3])"}</CodeBlock>

<Prose>{"These functions change the supplied tensor without recording the initialization in the autograd graph. PyTorch assumes the matrix will be used as "}<code>{"x @ weight.T"}</code>{", with shape "}<code>{"[fan_out, fan_in]"}</code>{". If your custom matrix is stored for "}<code>{"x @ weight"}</code>{" instead, initialize its transpose so the fan calculation corresponds to the operation. "}<a href={"https://docs.pytorch.org/docs/2.14/nn.init.html"}>{"PyTorch initialization API and orientation note"}</a>{""}</Prose>

<H2>{"Average preservation can hide a collapsed direction"}</H2>

<Prose>{"Imagine a circle of possible small changes to a two-dimensional input. Multiplying by a matrix turns it into an ellipse. The ellipse's longest and shortest radii are the matrix's "}<strong>{"singular values"}</strong>{": its strongest and weakest directional gains."}</Prose>

<Prose>{"Consider"}</Prose>

<div className="neural-equation"><MathBlock>{"M=\\begin{bmatrix}\\sqrt{1.9}&0\\\\0&\\sqrt{0.1}\\end{bmatrix}."}</MathBlock></div>

<Prose>{"The mean of its squared singular values is "}<InlineMath>{"(1.9+0.1)/2=1"}</InlineMath>{". Nevertheless, it stretches the horizontal direction by about 1.378 and shrinks the vertical one to about 0.316. Repeating this same matrix makes the discrepancy much larger. An average scale is not a guarantee about every direction."}</Prose>

<Prose>{"There is also a distinction between averaging over draws and inspecting one draw. For a square "}<InlineMath>{"d\\times d"}</InlineMath>{" matrix with independent, zero-mean entries of variance "}<InlineMath>{"1/d"}</InlineMath>{","}</Prose>

<div className="neural-equation"><MathBlock>{"E_W\\|Wx\\|^2=\\|x\\|^2"}</MathBlock></div>

<Prose>{"for each fixed "}<InlineMath>{"x"}</InlineMath>{". A particular sampled matrix need not preserve that norm. Our saved 64-by-64 Gaussian draw has singular values from approximately 0.00374 to 1.936."}</Prose>

<Prose>{"A square orthogonal "}<InlineMath>{"Q"}</InlineMath>{" satisfies "}<InlineMath>{"Q^\\top Q=I"}</InlineMath>{", so "}<InlineMath>{"\\|Qx\\|=\\|x\\|"}</InlineMath>{" for every "}<InlineMath>{"x"}</InlineMath>{". The program's QR construction checks this identity to maximum absolute error "}<InlineMath>{"8.9\\times10^{-16}"}</InlineMath>{" in float64. For a tall matrix, orthonormal columns can preserve input norms. For a wide matrix mapping to fewer dimensions, some input directions must be lost."}</Prose>

<Prose>{"The next operation still matters. For "}<InlineMath>{"f(x)=\\operatorname{ReLU}(\\sqrt2x)"}</InlineMath>{" at "}<InlineMath>{"x=(-1,1)"}</InlineMath>{","}</Prose>

<div className="neural-equation"><MathBlock>{"J_f=\\begin{bmatrix}0&0\\\\0&\\sqrt2\\end{bmatrix}."}</MathBlock></div>

<Prose>{"One local direction is completely blocked. The total input and output norms happen to agree at that particular point, which makes it an especially useful counterexample to judging the Jacobian from a single norm."}</Prose>

<InitializationGeometryLab />

<Prose>{"Keeping the singular values of a network's full input-output Jacobian close to one is the idea of "}<strong>{"dynamical isometry"}</strong>{". The linear and nonlinear cases require different conditions. Orthogonal initialization is useful evidence about an individual linear map, not a certificate that an entire ReLU network has this property. "}<a href={"https://arxiv.org/pdf/1312.6120"}>{"Saxe et al., dynamical-isometry discussion"}</a>{""}</Prose>

<InitializationSpectrumFigure />

<H2>{"Construct an orthogonal draw, then match a width-aware optimizer"}</H2>

<Prose>{"The scale formula tells us what to sample; an orthogonal initializer instead constrains a whole matrix. The complete "}<a href={"/learn-assets/weight-initialization-xavier-kaiming-p/initialization_library_bridge.py"}>{"initialization bridge"}</a>{" exposes both cases. For a matrix with more rows than columns, sample a Gaussian matrix, take reduced QR, and multiply each column of Q by the sign of the matching diagonal of R. The sign choice removes the QR routine's arbitrary sign convention. For more columns than rows, construct the tall counterpart and transpose it. Multiply by the requested gain last."}</Prose>

<CodeBlock language={"python"}>{"def orthogonal_matrix(rows, columns, gain=1.0, generator=None):\n    tall = torch.randn(max(rows, columns), min(rows, columns),\n                       dtype=torch.float64, generator=generator)\n    basis, triangular = torch.linalg.qr(tall, mode=\"reduced\")\n    signs = torch.where(triangular.diagonal() < 0, -1.0, 1.0)\n    basis = basis * signs\n    return gain * (basis if rows >= columns else basis.T)"}</CodeBlock>

<Prose>{"This uses the QR factorization already taught in "}<a href={"/learn/path/full-curriculum/matrix-decompositions-svd-qr-cholesky-lu"}>{"Matrix Decompositions"}</a>{"; it does not hide initialization inside "}<code>{"orthogonal_"}</code>{". With gain g, test "}<code>{"Q.T @ Q = g²I"}</code>{" for tall matrices and "}<code>{"Q @ Q.T = g²I"}</code>{" for wide ones. Testing the wrong identity would claim preservation in a dimension the map cannot preserve. Reduced QR costs O(max(m,n) min(m,n)²) arithmetic and O(mn) storage. This is a dense matrix construction, not a special accelerator kernel."}</Prose>

<Prose>{"Run "}<code>{"python initialization_library_bridge.py"}</code>{" with PyTorch. It compares rectangular Gram contracts against "}<code>{"nn.init.orthogonal_"}</code>{". Identical seeds need not yield identical wide matrices when routines draw arrays in different shapes; the meaningful comparison here is the distribution/construction contract and Gram identity."}</Prose>

<InitializationProgram file="initialization_library_bridge.py" title="Read the complete orthogonal and μP library bridge" />

<Prose>{"After the μP derivation below, continue with "}<code>{"python initialization_library_bridge.py --mup"}</code>{" in an environment containing Microsoft's "}<code>{"mup"}</code>{" package. That optional branch reuses "}<code>{"WidthMLP"}</code>{" directly from the adjacent experiment file without rerunning its training sweep. Its ordinary model substitutes "}<code>{"MuReadout"}</code>{", calls "}<code>{"set_base_shapes"}</code>{" with widths 32 and 64 to identify changing axes, copies the scratch model's already-parametrized weights using "}<code>{"rescale_params=False"}</code>{", and constructs "}<code>{"MuAdam"}</code>{". The input matrix has one changing axis, the hidden matrix two, and the readout one. Those annotations determine which optimizer group receives the width-divided rate. The readout performs the forward division."}</Prose>

<Prose>The Microsoft repository was marked archived on 21 September 2026. The pinned <code>mup==1.0.0</code> API is verified for this comparison; the reference link does not imply ongoing package maintenance. Preserve the tested environment and check compatibility before using a different version.</Prose>

<Prose>The supplied comparison uses widths 32 and 96, identical inputs and targets, zero initial optimizer state and two Adam updates. The executed run with PyTorch 2.14.0+cpu and Microsoft’s <code>mup==1.0.0</code> passes output, every parameter-gradient and updated-weight agreement at absolute/relative tolerances 1e−12. Turning parameter rescaling back on <strong>after</strong> copying the custom μP weights changes the experiment. <a href="https://github.com/microsoft/mup/blob/main/mup/layer.py">Readout</a>, <a href="https://github.com/microsoft/mup/blob/main/mup/shape.py">shape registration</a> and <a href="https://github.com/microsoft/mup/blob/main/mup/optim.py">optimizer source</a> expose the mapping. <a href={initializationAsset + 'native-verification.json'}>Executed environment and comparison record</a>.</Prose><Prose>Set up a separate environment with <code>python -m venv .venv</code>, activate it using your operating system’s command, then install <code>torch==2.14.0 numpy==2.3.5 scikit-learn==1.9.1 mup==1.0.0</code> with <code>python -m pip install</code>. Keep both Python files beside the CSV; the bridge imports <code>WidthMLP</code> without launching its training sweep.</Prose>

<Prose>{""}<strong>{"Implement a changed case."}</strong>{" Use a 3×7 orthogonal draw with gain 0.5 and change the μP target width to 160. Which identity and rates should the comparison check?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Distinguish output-row orthogonality from preserving every seven-dimensional input direction; the width ratio is measured against 32."}</Prose>

</details>

<details>

<summary>Solution and success criteria</summary>

<Prose>{"The Gram check is "}<code>{"Q @ Q.T = 0.25 I₃"}</code>{"; "}<code>{"Q.T @ Q"}</code>{" has rank at most three. The width ratio is five. For base Adam rate 0.003, input/readout rates remain 0.003, the hidden matrix rate becomes 0.0006, and the raw readout divides its input by five. Preserve the same copied state and compare two updates. A passing shape check alone does not establish those identities."}</Prose>

</details>

<H2>{"Why equal hidden units can stay equal"}</H2>

<Prose>{"Suppose two hidden units have identical incoming weights and biases, the same activation, and identical outgoing weights. Interchanging the units changes nothing. On the same example, they receive the same gradients, so an identical update preserves their equality. Two slots have learned one feature twice."}</Prose>

<Prose>{"In the saved two-unit example, input 1 passes through tanh, both incoming weights are 0.2, both outgoing weights are 0.3, the target is 1, and the loss is half squared error. Both incoming gradients are −0.25417. Changing the incoming weights to 0.1 and 0.3 gives gradients −0.26218 and −0.24234. Deliberately different deterministic values already break this symmetry; randomness is convenient, not logically necessary."}</Prose>

<Prose>{"“Never initialize anything to zero” is too broad. With distinct tanh features and a zero output head, the first hidden gradients are zero, but the head gradients are −0.09967 and −0.29131. The head can move first and open a later gradient route into the features. This is related to the zero-initialized LoRA output factor from the preceding lesson."}</Prose>

<Prose>{"An all-zero hidden ReLU network is a different case. Its features are zero, and PyTorch's ReLU derivative at zero is zero. The digit experiment below leaves such a model predicting equal classes. The useful question is which paths can learn on the first and subsequent updates."}</Prose>

<InitializationSymmetryLab />

<H2>{"Run a complete initialization comparison on real inputs"}</H2>

<Prose>{""}<a href={"/learn-assets/weight-initialization-xavier-kaiming-p/./initialization-experiments.py"}>{"Download the complete CPU program"}</a>{" together with "}<a href={"/learn-assets/weight-initialization-xavier-kaiming-p/./digits-400.csv"}>{"digits-400.csv"}</a>{". The "}<a href={"/learn-assets/weight-initialization-xavier-kaiming-p/./data-provenance.md"}>{"provenance and protocol"}</a>{" describe the real handwritten digits and exact split. The file contains 400 samples from UCI's optical digit dataset, not MNIST. Each 8-by-8 image becomes 64 inputs divided by the known feature-range maximum, 16."}</Prose>

<Prose>{"In an environment containing PyTorch, NumPy, and scikit-learn:"}</Prose>

<CodeBlock language={"text"}>{"python initialization-experiments.py"}</CodeBlock>

<Prose>{"The program includes imports, data loading, all model definitions, deterministic splits and initialization generators, optimization, metric calculations, and output recording. It needs no pretrained download. The prepared outputs were executed with Python 3.12, PyTorch 2.14.0+cpu, NumPy 2.3.5, and scikit-learn 1.9.1. Reproduction in another version can differ slightly."}</Prose>

<InitializationProgram /><div className="init-code-links"><a href={initializationAsset + "calculated-inputs.json"}>Complete measured record</a><a href={initializationAsset + "native-verification.json"}>Execution checks and environment</a></div>

<Prose>{"Read "}<code>{"DigitMLP"}</code>{" first. It has four 32-unit ReLU hidden layers and a ten-logit head. The six choices change hidden initialization; within each seed, all choices use the same randomly initialized head. All biases start at zero. The 280 training and 120 validation samples are identical across choices. Adam uses learning rate 0.003 for 300 full-batch updates."}</Prose>

<Prose>{""}<code>{"digit_fits"}</code>{" records cross-entropy and correct counts before training and after 1, 10, 100, and 300 updates. Cross-entropy measures assigned probability, while the correct count uses the largest logit. Neither alone describes every error."}</Prose>

<Prose>{"Actual seed-1 final results:"}</Prose>

<NeuralTable caption={"Run a complete initialization comparison on real inputs"} headers={[<>{"Hidden initialization"}</>,<>{"Training CE"}</>,<>{"Validation CE"}</>,<>{"Validation correct / 120"}</>]} rows={[[<>{"Zero"}</>,<>{"2.302586"}</>,<>{"2.302585"}</>,<>{"12"}</>],[<>{"Normal, std 0.01"}</>,<>{"0.002644"}</>,<>{"0.933934"}</>,<>{"106"}</>],[<>{"Xavier"}</>,<>{"0.000274"}</>,<>{"0.220745"}</>,<>{"116"}</>],[<>{"Kaiming"}</>,<>{"0.000303"}</>,<>{"0.175867"}</>,<>{"117"}</>],[<>{"Orthogonal, gain "}<InlineMath>{"\\sqrt2"}</InlineMath>{""}</>,<>{"0.000298"}</>,<>{"0.124550"}</>,<>{"117"}</>],[<>{"Normal, std 0.5"}</>,<>{"0.001471"}</>,<>{"0.778704"}</>,<>{"105"}</>]]} />

<Prose>{"Across three seeds, Kaiming gives 116–117 correct and orthogonal gives 116–118. Xavier gives 113–117. These small differences are not evidence for a universal winner. The much poorer small/large validation CE also shows why fitting the training set is not the same as assigning good probabilities on other examples."}</Prose>

<InitializationTrainingLab />

<Prose>{"The tiny initialization can eventually learn in this four-hidden-layer model. That does not contradict the twenty-layer signal probe: the architectures and questions differ."}</Prose>

<Prose>{""}<strong>{"Try a changed experiment:"}</strong>{" keep the split and seed fixed, change only the number of hidden layers, and record the initial signal statistics before training. Inspect how the changed depth alters the measured forward and backward statistics. Inspect both early optimization and final validation, and record your change as a new experiment rather than replacing the prepared observations. Repeated validation comparisons consume development information; this packet does not provide a final test estimate."}</Prose>

<H2>{"μP: changing width changes more than parameter count"}</H2>

<Prose>{"Suppose a learning rate worked well in a 32-unit network. Can you tune cheaply there and reuse it in a 128-unit network?"}</Prose>

<Prose>{"A forward sum of independent zero-mean random terms tends to grow on the order of the square root of their count. But a learning update is correlated with the inputs that produced its gradient. Summing those correlated changes can scale differently. Therefore an initialization that controls the first forward pass is not enough to control the size of the first learned feature change."}</Prose>

<Prose>{""}<strong>{"Maximal update parametrization"}</strong>{", written μP, coordinates initialization, forward multipliers, and optimizer scaling as width changes. Its associated μTransfer procedure tunes a smaller model and transfers eligible settings under the matching parametrization. It does not mean that every finite model has exactly the same best learning rate, nor that a rule derived for Adam can simply be copied into SGD. "}<a href={"https://arxiv.org/abs/2203.03466"}>{"μTransfer paper"}</a>{", "}<a href={"https://github.com/microsoft/mup"}>{"Microsoft's implementation and coordinate-check guide"}</a>{""}</Prose>

<Prose>{"Here is the exact restricted case in "}<code>{"WidthMLP"}</code>{": a bias-free 64→"}<InlineMath>{"n"}</InlineMath>{"→"}<InlineMath>{"n"}</InlineMath>{"→10 MLP with two ReLUs, fixed input/output sizes, base width "}<InlineMath>{"n_0=32"}</InlineMath>{", and width multiplier "}<InlineMath>{"m=n/n_0"}</InlineMath>{"."}</Prose>

<NeuralTable caption={"μP: changing width changes more than parameter count"} headers={[<>{"Component"}</>,<>{"Standard comparison"}</>,<>{"μP convention used here"}</>]} rows={[[<>{"Input weight std"}</>,<>{""}<InlineMath>{"\\sqrt{2/64}"}</InlineMath>{""}</>,<>{"same"}</>],[<>{"Hidden-to-hidden weight std"}</>,<>{""}<InlineMath>{"\\sqrt{2/n}"}</InlineMath>{""}</>,<>{"same"}</>],[<>{"Raw readout weight std"}</>,<>{""}<InlineMath>{"1/\\sqrt n"}</InlineMath>{""}</>,<>{""}<InlineMath>{"1/\\sqrt{n_0}"}</InlineMath>{""}</>],[<>{"Readout input"}</>,<>{""}<InlineMath>{"h"}</InlineMath>{""}</>,<>{""}<InlineMath>{"h/m"}</InlineMath>{""}</>],[<>{"Adam input learning rate"}</>,<>{""}<InlineMath>{"\\eta"}</InlineMath>{""}</>,<>{""}<InlineMath>{"\\eta"}</InlineMath>{""}</>],[<>{"Adam hidden learning rate"}</>,<>{""}<InlineMath>{"\\eta"}</InlineMath>{""}</>,<>{""}<InlineMath>{"\\eta/m"}</InlineMath>{""}</>],[<>{"Adam raw-readout learning rate"}</>,<>{""}<InlineMath>{"\\eta"}</InlineMath>{""}</>,<>{""}<InlineMath>{"\\eta"}</InlineMath>{""}</>]]} />

<Prose>{"The raw μP readout is deliberately distinguished from its "}<strong>{"effective"}</strong>{" multiplication by "}<InlineMath>{"1/m"}</InlineMath>{". Leaving out that multiplier changes the model. At the base width "}<InlineMath>{"m=1"}</InlineMath>{", both columns describe exactly the same training procedure."}</Prose>

<Prose>{"A readout has one dimension that grows with width; the middle matrix has two. The package calls these its infinite dimensions because it tracks what happens as width grows. This table follows the corresponding readout and hidden-matrix conventions of "}<code>{"MuReadout"}</code>{" and "}<code>{"MuAdam"}</code>{" for this architecture. The program makes the rules explicit to keep the example runnable without another dependency. For tied parameters, multiple changing dimensions, biases, attention, a different optimizer, or an existing architecture, use and inspect the package's base-shape machinery rather than treating this class as a generic replacement. "}<a href={"https://github.com/microsoft/mup/blob/main/mup/layer.py"}>{"Readout implementation"}</a>{", "}<a href={"https://github.com/microsoft/mup/blob/main/mup/optim.py"}>{"optimizer implementation"}</a>{""}</Prose>

<Prose>{"The complete experiment fits widths 32, 64, and 128 with learning rates 0.001, 0.003, and 0.01, for both parametrizations and three seeds. Each fit uses 150 full-batch updates on the same digit split. These are 54 small CPU fits."}</Prose>

<Prose>{"Mean final validation cross-entropy across the three seeds:"}</Prose>

<NeuralTable caption={"μP: changing width changes more than parameter count"} headers={[<>{"Parametrization"}</>,<>{"Width"}</>,<>{"LR 0.001"}</>,<>{"LR 0.003"}</>,<>{"LR 0.01"}</>]} rows={[[<>{"Standard"}</>,<>{"32"}</>,<>{"0.25880"}</>,<>{""}<strong>{"0.10939"}</strong>{""}</>,<>{"0.11739"}</>],[<>{"Standard"}</>,<>{"64"}</>,<>{"0.12406"}</>,<>{""}<strong>{"0.10485"}</strong>{""}</>,<>{"0.12290"}</>],[<>{"Standard"}</>,<>{"128"}</>,<>{""}<strong>{"0.08466"}</strong>{""}</>,<>{"0.08567"}</>,<>{"0.12092"}</>],[<>{"μP"}</>,<>{"32"}</>,<>{"0.25880"}</>,<>{""}<strong>{"0.10939"}</strong>{""}</>,<>{"0.11739"}</>],[<>{"μP"}</>,<>{"64"}</>,<>{"0.20471"}</>,<>{""}<strong>{"0.09638"}</strong>{""}</>,<>{"0.11020"}</>],[<>{"μP"}</>,<>{"128"}</>,<>{"0.19483"}</>,<>{""}<strong>{"0.08208"}</strong>{""}</>,<>{"0.09354"}</>]]} />

<Prose>{"The narrow-model choice, 0.003, remains best among these three candidates at the larger μP widths. In the standard width-128 run, 0.001 has slightly lower average CE than 0.003; the difference is only about 0.00101. This is an illustrative local experiment, not proof of exact transfer or a discovery of the global optimum. It also does not establish a compute or memory speedup."}</Prose>

<Prose>{"A "}<strong>{"coordinate check"}</strong>{" inspects activation magnitudes as width changes, before spending a long run on tuning. The program records mean absolute values of five corresponding tensors at steps 0, 1, 2, 5, and 150 on the same 32 training examples. Compare the same tensor at the same step."}</Prose>

<Prose>{"For seed 1 and learning rate 0.01, the μP mean absolute output at initialization is 0.381, 0.308, and 0.170 for widths 32, 64, and 128. It is not flat. A nonzero random μP readout can have a decaying initial output scale before correlated updates develop. The official guide discusses this transient; demanding exact equality would reject a valid behavior."}</Prose>

<InitializationWidthLab />

<H2>{"Deeper tools and practical failure checks"}</H2>

<Prose>{""}<strong>{"Data-dependent initialization."}</strong>{" LSUV starts with orthogonal weights, sends a calibration batch through successive layers, and rescales each layer toward a chosen output variance. This adapts to an observed distribution rather than only an assumed one. The procedure is bounded by a tolerance and maximum iteration count. A zero or tiny variance needs a diagnostic stop, not division by zero. Calibration on training inputs is legitimate; using a held-out evaluation distribution to fit those scales consumes that information. The chosen measurement point—before or after an activation—must be explicit. "}<a href={"https://arxiv.org/pdf/1511.06422"}>{"LSUV, Algorithm 1"}</a>{""}</Prose>

<Prose>{""}<strong>{"Residual initialization."}</strong>{" A residual block computes "}<InlineMath>{"x+F(x)"}</InlineMath>{". Making "}<InlineMath>{"F"}</InlineMath>{" initially small can begin near a usable identity map. Fixup uses a coordinated recipe including depth-scaled earlier branch weights, zero final branch/classification layers, and specified scalar multipliers and biases. For a branch containing "}<InlineMath>{"r"}</InlineMath>{" weight layers, its earlier branch weights use a factor "}<InlineMath>{"L^{-1/(2r-2)}"}</InlineMath>{", where "}<InlineMath>{"L"}</InlineMath>{" is the number of residual branches. This is not a universal instruction to multiply any second layer by "}<InlineMath>{"1/\\sqrt L"}</InlineMath>{". The next lesson makes the paths and placement concrete. "}<a href={"https://arxiv.org/pdf/1901.09321"}>{"Fixup, §3"}</a>{""}</Prose>

<Prose>{""}<strong>{"Truncated normal is not clipping."}</strong>{" Values outside the interval are redrawn; they do not pile up at the endpoints. In "}<code>{"trunc_normal_"}</code>{", bounds are absolute values. With "}<code>{"std=0.02"}</code>{", the default bounds −2 and 2 are one hundred standard deviations away. To truncate at two standard deviations, specify −0.04 and 0.04. In the program's 100,000-value float64 sample, these choices produce standard deviations about 0.02000 and 0.01758 respectively. The truncated distribution has less variance; the function does not silently restore 0.02 afterward. "}<a href={"https://docs.pytorch.org/docs/2.14/nn.init.html#torch.nn.init.trunc_normal_"}>{"PyTorch truncated-normal contract"}</a>{""}</Prose>

<Prose>{""}<strong>{"Small numbers and precision."}</strong>{" BF16's reduced precision does not mean every number below 0.001 becomes zero. It has a broad exponent range. In our conversion fixture, "}<InlineMath>{"10^{-5}"}</InlineMath>{", "}<InlineMath>{"10^{-6}"}</InlineMath>{", and "}<InlineMath>{"10^{-8}"}</InlineMath>{" all remain nonzero in BF16. The last becomes zero in float16. Loss of a small update when added to a much larger weight is a different issue from representing the small value by itself. Measure the operation you are concerned about."}</Prose>

<InitializationPrecisionFigure />

<Prose>{""}<strong>{"Normalization and defaults."}</strong>{" Normalization can change activation scale, but its axes, epsilon, affine parameters, and Jacobian still matter. It does not certify that all initial gradients or update sizes are useful. Inspect the actual module rather than relying on a catalogue of alleged architecture defaults: for example, PyTorch's LSTM documents a hidden-size-based uniform initialization for all its weights and biases. It does not promise a default orthogonal recurrent matrix. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html"}>{"LSTM initialization note"}</a>{""}</Prose>

<Prose>{"When a model fails, inspect initial preactivations, post-activation mean square, zero/saturated fractions, gradient magnitudes, and actual update-to-weight magnitudes. Check inputs, loss, and labels too. Large logits do not automatically make a stable cross-entropy implementation overflow; large finite losses, saturation elsewhere, and excessive parameter updates are separate diagnoses."}</Prose>

<H2>{"Practice: use the mechanism on a changed case"}</H2>

<H3>{"1. Initialize a wider-input layer"}</H3>

<Prose>{"A ReLU layer has 200 inputs and 50 outputs. Find the fan-in Kaiming normal standard deviation and uniform bounds. Then calculate Xavier's idealized linear forward and backward factors."}</Prose>

<details><summary>Hint</summary>

<Prose>{"square the standard deviation to obtain variance; uniform variance is bound squared divided by three."}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{"Kaiming variance is "}<InlineMath>{"2/200=0.01"}</InlineMath>{", so std is 0.1 and bounds are "}<InlineMath>{"\\pm\\sqrt{0.03}\\approx\\pm0.1732"}</InlineMath>{". Xavier variance is "}<InlineMath>{"2/250=0.008"}</InlineMath>{", giving forward factor 1.6 and backward factor 0.4. Fan-out Kaiming would be a different priority, with variance "}<InlineMath>{"2/50=0.04"}</InlineMath>{"."}</Prose>

</details>

<H3>{"2. Repair a misleading statistic"}</H3>

<Prose>{"A layer's pooled mean square stays at 1, while its pooled mean changes from 0 to 0.8. Someone reports “the variance stayed at 1.” Correct the report."}</Prose>

<details><summary>Hint</summary>

<Prose>{"subtract the squared mean."}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{"the final variance is "}<InlineMath>{"1-0.8^2=0.36"}</InlineMath>{". The second moment stayed constant. We also need to know whether the samples and coordinates were pooled consistently before comparing those measurements."}</Prose>

</details>

<H3>{"3. Keep average gain, lose a direction"}</H3>

<Prose>{"Construct a diagonal 2-by-2 matrix whose average squared singular value is 1 but whose smaller singular value is 0.2. Find the larger singular value and the gain after five repetitions along the smaller direction."}</Prose>

<details><summary>Hint</summary>

<Prose>{"the two squared singular values must sum to 2."}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{"the larger is "}<InlineMath>{"\\sqrt{1.96}=1.4"}</InlineMath>{". The smaller direction has gain "}<InlineMath>{"0.2^5=0.00032"}</InlineMath>{". Preserving the average does not protect this direction."}</Prose>

</details>

<H3>{"4. Follow the first update"}</H3>

<Prose>{"Two tanh hidden units have different incoming weights, but their outgoing weights are both zero. Is there no learning signal anywhere? Contrast this with an all-zero hidden ReLU network."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Trace the chain rule from the output backward, using the actual hidden features at each parameter."}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{"the output weights can have nonzero gradients because the hidden features differ and are nonzero; earlier weights receive no gradient through the zero output weights on that first step. After the output moves, that route can open. In the all-zero ReLU case used here, the features and their chosen derivatives at zero block the relevant paths. Zero initialization must be assessed by location."}</Prose>

</details>

<H3>{"5. Transfer a base learning rate carefully"}</H3>

<Prose>{"For the exact bias-free μP model above, base width is 32, target width is 256, and base Adam learning rate is 0.004. Specify the three group rates, raw readout std, and forward divisor."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Find the width ratio, then follow the separate input, hidden-matrix and readout conventions."}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{""}<InlineMath>{"m=8"}</InlineMath>{". Input and raw-readout rates are 0.004; the hidden matrix rate is 0.0005. Raw readout std remains "}<InlineMath>{"1/\\sqrt{32}"}</InlineMath>{"; divide its input by 8. These answers depend on the stated Adam parametrization, not a general rule for all optimizers."}</Prose>

</details>

<H3>{"6. Plan a useful failure investigation"}</H3>

<Prose>{"You observe a roughly constant forward mean square but a tiny gradient in an early layer. Propose two checks that distinguish explanations."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Ask whether an average scale conceals different directions, and whether the current activations pass the incoming gradient."}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{"inspect the local Jacobian or directional sensitivities to test whether some directions are blocked despite average scale preservation. Separately inspect activation gates/saturation and the backward signal arriving from the head. Record which scalar output or loss supplied that gradient. A single final norm cannot distinguish all these cases."}</Prose>

</details>

<H2>{"Another way to learn, and what comes next"}</H2>

<ul><li>{""}<a href={"https://www.youtube.com/watch?v=wEoyxE0GP2M"}>{"Stanford CS231n, Lecture 6: Training Neural Networks I"}</a>{" offers a lecture-based route through activation, initialization, and normalization. The official "}<a href={"https://cs231n.stanford.edu/2017/syllabus"}>{"2017 syllabus"}</a>{" identifies its scope. Use it for intuition; this lesson's μP material and versioned API checks go beyond that lecture."}</li><li>{""}<a href={"https://www.microsoft.com/en-us/research/blog/%C2%B5transfer-a-technique-for-hyperparameter-tuning-of-enormous-neural-networks/"}>{"Microsoft Research's μTransfer article"}</a>{" explains why random forward sums and correlated training updates require different reasoning. It is a historical 2022 introduction, not a current catalogue of every supported model."}</li><li>{""}<a href={"https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"}>{"The original Xavier paper"}</a>{" is useful after the local calculation: read the assumptions and compare its activation diagnostics with the measurements here."}</li><li>{""}<a href={"https://github.com/microsoft/mup"}>{"The μP repository"}</a>{" is the implementation route after the restricted example. Study base shapes, readout layers, optimizer choice, and coordinate checks together."}</li></ul>

<Prose>{"The next topic in this module is "}<a href={"/learn/path/full-curriculum/residual-connections-skip-connections?module=deep-learning-fundamentals"}>{"Residual Connections & Skip Connections"}</a>{". Initialization controls the starting transformations. A residual path changes how those transformations are connected, giving the network a direct route alongside the learned correction."}</Prose>
</div>
};
