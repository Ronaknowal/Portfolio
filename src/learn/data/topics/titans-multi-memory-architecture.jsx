// Generated from the complete twelve-section prepared manuscript and all changed practice.
import {Prose,H2,H3,CodeBlock} from '../../components/content';
import {Math as InlineMath,MathBlock} from '../../components/content/Math.jsx';
import {NeuralTable} from '../../components/lesson-labs/NeuralLessonElements.jsx';
import {TitansWriteLab,TitansGatedLab,TitansChunkLab,TitansOuterLab} from '../../components/lesson-labs/TitansMemoryLabs.jsx';
import {TitansLifetimesFigure,TitansWriteFigure,TitansUpdateFigure,TitansStabilityFigure,TitansNonlinearFigure,TitansProjectionFigure,TitansJunctionsFigure,TitansOuterFigure,TitansTimelineFigure,TitansPayloadFigure,TitansScanFigure,TitansResearchFigure} from '../../components/lesson-labs/TitansMemoryDiagrams.jsx';
import {TitansPairedFigure,TitansRentalLab,TitansProgram,TitansDownloads} from '../../components/lesson-labs/TitansMemoryStudy.jsx';
export default {title:'Titans: Writing into a Multi-Memory Architecture',readTime:'~90 min read + investigations and practice',content:()=> <div className="neural-lesson neural-lesson-neutral titans-lesson">
<Prose opening="exploration">{""}<strong>{"Explore as you read."}</strong>{" Write a new association and watch how it changes a later query. Then follow the same tokens through attention and neural memory, compare two update schedules, and inspect when real rental observations become available for adaptation."}</Prose>

<Prose>{"Imagine reading a long maintenance log. You need the last few entries to understand what is happening now, an impression of older recurring faults, and general knowledge about how maintenance reports are written. Keeping every entry immediately accessible costs space. Compressing everything into one small summary risks losing a detail you will need later."}</Prose>

<Prose>{"Titans explores a combination: attention over recent context, a small neural network whose weights change as it processes the sequence, and learned vectors that carry information shared across sequences. The unusual part is the second one. Reading this memory means running a neural network. Writing to it means taking a gradient step on that network."}</Prose>

<Prose opening="route">{""}<strong>{"First-pass route."}</strong>{" Start with §§ 1–4 and the associative-memory investigation: follow two writes by hand before reading the code. Then use the three-branch example in § 5, run the real-data program in § 7, and attempt practice 1–4. Return for the outer derivative in § 6, chunk parallelization in § 8, state accounting in § 9 and the research-reading branch in § 10. Expect roughly 40–50 minutes of core reading, with a separate 45–75 minutes for the first exercises and programs. The deeper branches can be another sitting."}</Prose>

<Prose opening="prerequisites">{"This is an architecture lesson, so a little neural-network background helps. A "}<strong>{"weight"}</strong>{" is a number used by a model; a "}<strong>{"loss"}</strong>{" measures an error; a "}<strong>{"gradient"}</strong>{" tells us how changing the weights changes that loss. A "}<strong>{"query"}</strong>{" asks for a representation, a "}<strong>{"key"}</strong>{" identifies an association, and a "}<strong>{"value"}</strong>{" is the representation associated with that key. We will make those roles concrete before using large tensors. The previous "}<a href={"/learn/path/full-curriculum/hybrid-ssm-transformer-architectures-jamba?module=deep-learning-fundamentals"}>{"Jamba lesson"}</a>{" combines attention and recurrence. Here, the recurrent state becomes the parameters of a learner."}</Prose>

<H2>{"1. Three places information can live"}</H2>

<Prose>{"Think of a workbench, an adjustable prediction rule, and a shared instruction card. The workbench keeps recent items available; the rule changes with experience; the instruction card begins the same for each new job. The analogy describes roles. It does not establish that these components implement human memory or keep reliable copies of arbitrary facts."}</Prose>

<NeuralTable caption={"1. Three places information can live"} headers={[<>{"Component"}</>,<>{"What is stored?"}</>,<>{"What changes during one sequence?"}</>,<>{"How is it used?"}</>]} rows={[[<>{"Recent attention context"}</>,<>{"Representations of recent positions, often cached as projected keys and values"}</>,<>{"Positions enter and leave the permitted window"}</>,<>{"A query forms a weighted combination of accessible values"}</>],[<>{"Contextual neural memory"}</>,<>{"Fast weights and their update state"}</>,<>{"Gradient-based writes change the weights and momentum"}</>,<>{"A query passes through the current memory network"}</>],[<>{"Persistent memory"}</>,<>{"Learned input-independent vectors"}</>,<>{"Their shared parameter values stay fixed at inference"}</>,<>{"The core can attend to these prefix representations"}</>]]} />

<TitansLifetimesFigure/>

<Prose>{"An attention cache also stores transformed representations, not a literal database of original text. Its advantage is that separate positions remain addressable inside its permitted context. Neural memory compresses associations into shared parameters: a later write can alter more than one earlier answer. Its storage can stay bounded while sequence length grows; its information capacity remains finite."}</Prose>

<Prose>{"Persistent vectors are not the same as the initial fast weights. A memory can start from nonzero, learned or otherwise specified initial weights even when no persistent prefix is present. Removing the prefix therefore does not logically imply that the model starts with no useful prior. What a trained prefix learns must be assessed through the task, not inferred from its name."}</Prose>

<H2>{"2. Write a rule instead of appending a record"}</H2>

<Prose>{"Start with a memory that accepts a two-number key and returns one number:"}</Prose>

<div className="neural-equation"><MathBlock>{"M_w(k)=w_1 k_1+w_2 k_2."}</MathBlock></div>

<Prose>{"The key might represent two features of an event. For the arithmetic example, the numbers are deliberately constructed. Our first desired association is key "}<InlineMath>{"(1,0)"}</InlineMath>{" → value 2. Starting from "}<InlineMath>{"w=(0,0)"}</InlineMath>{", the memory answers 0. Define the residual as prediction minus desired value, so it is −2."}</Prose>

<Prose>{"We use "}<strong>{"half-squared error"}</strong>{" throughout the calculations:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\ell(w;k,v)=\\tfrac12(M_w(k)-v)^2,\n\\qquad g=\\nabla_w\\ell=(M_w(k)-v)k."}</MathBlock></div>

<Prose>{"The one-half cancels the factor 2 when differentiating a square. The main Titans objective writes squared error without the half; its appendix uses the half convention. To obtain the same step when switching to the unhalved loss, halve the learning rate. Loss reduction and learning rate belong together."}</Prose>

<Prose>{"Here "}<InlineMath>{"g=(-2,0)"}</InlineMath>{". A gradient step with rate "}<InlineMath>{"\\theta=0.5"}</InlineMath>{" gives"}</Prose>

<div className="neural-equation"><MathBlock>{"w_{\\mathrm{new}}=w-\\theta g=(0,0)-0.5(-2,0)=(1,0)."}</MathBlock></div>

<Prose>{"Querying "}<InlineMath>{"(1,0)"}</InlineMath>{" again now returns 1. One update moved the answer toward 2. It did not create a perfect database entry."}</Prose>

<TitansWriteFigure/>

<H3>{"The second write reveals interference"}</H3>

<Prose>{"Keep the same weights "}<InlineMath>{"(1,0)"}</InlineMath>{", rate 0.5 and a new target 4. First use the key "}<InlineMath>{"(0,1)"}</InlineMath>{". The prediction is 0, residual −4, gradient "}<InlineMath>{"(0,-4)"}</InlineMath>{", and new weights "}<InlineMath>{"(1,2)"}</InlineMath>{". The original query "}<InlineMath>{"(1,0)"}</InlineMath>{" still returns 1."}</Prose>

<Prose>{"Now restart from "}<InlineMath>{"(1,0)"}</InlineMath>{" and use "}<InlineMath>{"(1,1)"}</InlineMath>{" as the second key. The prediction is 1, residual −3, gradient "}<InlineMath>{"(-3,-3)"}</InlineMath>{", and new weights "}<InlineMath>{"(2.5,1.5)"}</InlineMath>{". The original query now returns 2.5. The new association changed an old answer because the keys share a direction in parameter space."}</Prose>

<NeuralTable caption={"The second write reveals interference"} headers={[<>{"Second key"}</>,<>{"New weights"}</>,<>{"Answer to original key"}</>,<>{"Interpretation"}</>]} rows={[[<>{""}<InlineMath>{"(0,1)"}</InlineMath>{""}</>,<>{""}<InlineMath>{"(1,2)"}</InlineMath>{""}</>,<>{"1"}</>,<>{"Orthogonal direction leaves this old read unchanged"}</>],[<>{""}<InlineMath>{"(1,1)"}</InlineMath>{""}</>,<>{""}<InlineMath>{"(2.5,1.5)"}</InlineMath>{""}</>,<>{"2.5"}</>,<>{"Correlated direction changes the old read"}</>]]} />

<Prose>{"You can predict this without recalculating every weight. Let "}<InlineMath>{"q"}</InlineMath>{" be the old query and "}<InlineMath>{"r=M_w(k)-v"}</InlineMath>{" the new residual. For one linear update without decay or momentum,"}</Prose>

<div className="neural-equation"><MathBlock>{"\\Delta M(q)=-\\theta r\\,k^\\top q."}</MathBlock></div>

<Prose>{"The change is governed by the overlap "}<InlineMath>{"k^\\top q"}</InlineMath>{". This is the same old-read difference seen in the table, reached by an algebraic route. Orthogonal keys give zero overlap; other keys may reinforce or disrupt an old answer. In a nonlinear memory, local parameter sensitivities replace this simple key-overlap calculation."}</Prose>

<TitansWriteLab/>

<H2>{"3. Error, gradient, momentum and forgetting are different quantities"}</H2>

<Prose>{"With many output coordinates, write "}<InlineMath>{"M_\\phi(k)\\in\\mathbb R^{d_v}"}</InlineMath>{", where "}<InlineMath>{"\\phi"}</InlineMath>{" contains all memory weights. The residual is "}<InlineMath>{"r=M_\\phi(k)-v"}</InlineMath>{". If "}<InlineMath>{"J"}</InlineMath>{" is the matrix of derivatives of the memory output with respect to its parameters, the half-squared-loss gradient is"}</Prose>

<div className="neural-equation"><MathBlock>{"g=J^\\top r."}</MathBlock></div>

<Prose>{"A large residual can produce a large gradient, but the parameter sensitivity matters. In our bias-free linear memory, the key "}<InlineMath>{"(0,0)"}</InlineMath>{" gives zero gradient even if the target is 10 and the loss is 50: multiplying any weights by that key still gives 0. Changing those weights cannot fix this example. A bias or different key representation changes the situation."}</Prose>

<Prose>{"The Titans paper motivates writes through “surprise.” For implementation, keep the following measurements separate: loss, parameter-gradient norm, momentum, actual parameter change and later task performance. None is a direct measurement of a fact's semantic importance. In particular, a factor 4 difference in squared loss implies a factor 2 difference in residual norm; it does not generally imply a factor 2 gradient norm, even for a fixed Jacobian, because residual directions can differ."}</Prose>

<H3>{"Add a memory of recent updates"}</H3>

<Prose>{"Let "}<InlineMath>{"S"}</InlineMath>{" have the same shapes as the memory weights. One update is"}</Prose>

<div className="neural-equation"><MathBlock>{"g_t=\\nabla_{\\phi}\\ell(\\phi_{t-1};k_t,v_t),\\qquad\nS_t=\\eta_t S_{t-1}-\\theta_t g_t,"}</MathBlock></div>

<div className="neural-equation"><MathBlock>{"\\phi_t=(1-\\alpha_t)\\phi_{t-1}+S_t."}</MathBlock></div>

<Prose>{"Read the equations in this order: compute today's gradient at the old weights; retain a fraction of the previous update; add today's gradient step; shrink the old weights; add the resulting update. "}<InlineMath>{"\\theta_t"}</InlineMath>{" controls the new gradient, "}<InlineMath>{"\\eta_t"}</InlineMath>{" retains momentum, and "}<InlineMath>{"\\alpha_t"}</InlineMath>{" controls direct shrinkage. Titans makes these quantities data-dependent through learned mappings. Fixed values in our calculations let us isolate the mechanism."}</Prose>

<Prose>{"For one weight, take "}<InlineMath>{"w=1"}</InlineMath>{", old "}<InlineMath>{"S=0.2"}</InlineMath>{", "}<InlineMath>{"g=-2"}</InlineMath>{", rate 0.1, momentum retention 0.5 and decay 0.1. Then "}<InlineMath>{"S'=0.5(0.2)-0.1(-2)=0.3"}</InlineMath>{", and "}<InlineMath>{"w'=0.9(1)+0.3=1.2"}</InlineMath>{". The weight changed by 0.2, while the momentum is 0.3. Decay accounts for the difference."}</Prose>

<TitansUpdateFigure/>

<Prose>{"If the new gradient is zero, existing momentum can still move the weights. If "}<InlineMath>{"\\alpha=1"}</InlineMath>{", the retained old-weight term vanishes, but "}<InlineMath>{"S_t"}</InlineMath>{" can remain nonzero. A full sequence reset therefore restores the specified initial weights and clears momentum, recent context, convolution history and position/chunk state. Setting one forget gate to 1 is not a complete reset."}</Prose>

<Prose>{"With no writes and no momentum, repeated constant decay gives "}<InlineMath>{"\\phi_t=(1-\\alpha)^t\\phi_0"}</InlineMath>{". For "}<InlineMath>{"\\alpha=0.01"}</InlineMath>{", half the initial amplitude remains after about 68.97 steps. This is a decay calculation, not a measured lifetime for stored facts: continued writes and nonlinear readouts change actual recall."}</Prose>

<H3>{"When can a simple write diverge?"}</H3>

<Prose>{""}<strong>{"Deeper branch."}</strong>{" For one fixed key in linear memory, with no momentum or decay, the residual after a step is"}</Prose>

<div className="neural-equation"><MathBlock>{"r'=r(1-\\theta\\|k\\|^2)."}</MathBlock></div>

<Prose>{"Thus repeated updates reduce the residual magnitude when "}<InlineMath>{"0<\\theta\\|k\\|^2<2"}</InlineMath>{". At 1, this particular association is fitted in one step. At 2, the residual alternates sign without shrinking. Above 2, its magnitude grows. This explains why key scale and learning rate must be considered together. It also explains why the zero-key example cannot improve."}</Prose>

<TitansStabilityFigure/>

<Prose>{"A nonlinear network, changing keys and momentum add further dependencies. Decay is neither necessary for every stable sequence nor sufficient to rescue every unstable one. Monitor finite values, gradients and updates; inspect their cause before changing a rate. Clipping is an explicit modification to the update rule, with a specified norm and threshold, rather than a universal default supplied by the architecture's name."}</Prose>

<H2>{"4. Make the memory nonlinear"}</H2>

<Prose>{"A two-layer memory can map a key through hidden features:"}</Prose>

<div className="neural-equation"><MathBlock>{"M_\\phi(k)=W_2\\,\\operatorname{SiLU}(W_1k+b_1)+b_2."}</MathBlock></div>

<Prose>{"Here "}<InlineMath>{"W_1"}</InlineMath>{" has shape "}<InlineMath>{"H\\times d_k"}</InlineMath>{", "}<InlineMath>{"b_1"}</InlineMath>{" has "}<InlineMath>{"H"}</InlineMath>{" entries, "}<InlineMath>{"W_2"}</InlineMath>{" has shape "}<InlineMath>{"d_v\\times H"}</InlineMath>{", and "}<InlineMath>{"b_2"}</InlineMath>{" has "}<InlineMath>{"d_v"}</InlineMath>{" entries. SiLU maps a scalar "}<InlineMath>{"a"}</InlineMath>{" to "}<InlineMath>{"a\\sigma(a)"}</InlineMath>{", where "}<InlineMath>{"\\sigma"}</InlineMath>{" is the logistic sigmoid. The intermediate "}<InlineMath>{"H"}</InlineMath>{"-vector contains nonlinear combinations of the key coordinates. We can still compute a residual, backpropagate it and update every parameter."}</Prose>

<TitansNonlinearFigure/>

<Prose>{"Why add depth? A linear rule cannot express every relationship between keys and values. For example, a single affine function cannot map both "}<InlineMath>{"(1,1)"}</InlineMath>{" and "}<InlineMath>{"(-1,-1)"}</InlineMath>{" to 1 while mapping "}<InlineMath>{"(1,-1)"}</InlineMath>{" and "}<InlineMath>{"(-1,1)"}</InlineMath>{" to −1. Adding the two positive-case equations fixes twice the bias at 2; adding the negative-case equations fixes it at −2, a contradiction. Nonlinear hidden features can distinguish these patterns."}</Prose>

<Prose>{"More expressive functions are useful, but there is no general law saying “H hidden units store H independent facts.” Capacity also depends on precision, key geometry, objectives, training and the tolerated retrieval error. Repeatedly rehearsing the same examples and processing each example once are different experiments."}</Prose>

<H3>{"A complete memory cell you can inspect"}</H3>

<Prose>{"The supplied "}<a href={"/learn-assets/titans-multi-memory-architecture/neural_memory.py"}>{"neural_memory.py"}</a>{" contains the forward function, initialization, parameter copying and this update. It uses a scalar value per key; a batch averages the half-squared scalar losses. Each request owns its parameter and momentum tuples. The update returns new tensors instead of silently changing a shared global model."}</Prose>

<TitansProgram file="neural_memory.py"/>

<CodeBlock language={"python"}>{"import torch\nfrom neural_memory import initialize_memory, read_memory, write_memory\n\nparameters = initialize_memory(seed=3, input_size=2, hidden_size=3)\nmomentum = tuple(torch.zeros_like(weight) for weight in parameters)\nkey = torch.tensor([1.0, -0.5], dtype=torch.float64)\nvalue = torch.tensor(0.8, dtype=torch.float64)\n\nparameters, momentum, loss, gradients = write_memory(\n    parameters, momentum, key, value,\n    rate=0.1, retention=0.0, decay=0.0,\n    differentiable=True,\n)\nquery = torch.tensor([-0.25, 0.75], dtype=torch.float64)\nretrieved = read_memory(parameters, query)"}</CodeBlock>

<Prose>{"The query differs from the write key. This matters: making the training association easier does not mean every possible query becomes more useful. The next section gives attention another path to the information it needs; § 6 explains why the outer task trains the system to make the paths useful together."}</Prose>

<H2>{"5. Connect memory to attention"}</H2>

<Prose>{"Titans describes three wiring choices. The distinction is where a memory read enters the computation, not a fixed mapping from task names to a universally best variant. The paper's equations omit some residual and normalization detail, so these diagrams describe their main data paths."}</Prose>

<Prose>{"In a full model, the input representation "}<InlineMath>{"x_t\\in\\mathbb R^{d_{in}}"}</InlineMath>{" supplies learned views. With our column-vector convention, "}<InlineMath>{"k_t=W_Kx_t"}</InlineMath>{", "}<InlineMath>{"v_t=W_Vx_t"}</InlineMath>{", and "}<InlineMath>{"q_t=W_Qx_t"}</InlineMath>{". The projection shapes are "}<InlineMath>{"d_k\\times d_{in}"}</InlineMath>{", "}<InlineMath>{"d_v\\times d_{in}"}</InlineMath>{", and "}<InlineMath>{"d_k\\times d_{in}"}</InlineMath>{", respectively. The write learns to associate the key view with the value view; the query view chooses what to read. The projection weights belong to the outer training loop, while the contextual memory changes within a sequence. Input-conditioned rate and retention mappings can also learn which updates are useful."}</Prose>

<Prose>{"The paper's fuller blocks use residual connections, normalized queries/keys, nonlinear projections and short depthwise convolutions. A causal convolution mixes a channel's recent positions with learned coefficients, giving a key some local history before the memory sees it. That history is another piece of sequence state to carry across a boundary. Normalizing a nonzero key controls its length, which connects directly to the key-scale effect derived in § 3. Our unnormalized linear investigation keeps the zero key valid so you can inspect that boundary explicitly."}</Prose>

<NeuralTable caption={"5. Connect memory to attention"} headers={[<>{"Variant"}</>,<>{"Main path"}</>,<>{"What attention receives"}</>]} rows={[[<>{"Memory as Context, MAC"}</>,<>{"Read historical memory for a segment → append retrieved context and persistent prefix → attention → update memory from the resulting representation → gated output"}</>,<>{"The segment plus retrieved historical representations and persistent prefix"}</>],[<>{"Memory as Gate, MAG"}</>,<>{"Persistent prefix and sequence feed a local attention branch and a neural-memory branch → combine their outputs through a nonlinear gate"}</>,<>{"Prefix and accessible recent input representations"}</>],[<>{"Memory as Layer, MAL"}</>,<>{"Prefix and input → memory layer → attention layer"}</>,<>{"The memory layer's transformed sequence"}</>]]} />

<TitansJunctionsFigure/>

<Prose>{"In MAC, current input queries the historical state before the segment update. The attention result then supplies information for writing. In the paper's schematic, the final output also uses a read from the updated memory. A write followed by a read is consequently not automatically a causality error. The question is which observations were available to produce that write."}</Prose>

<Prose>{"For autoregressive prediction at position t, every dependency must originate in tokens already observed at that position. A retrieved vector depends on its query as well as on the old memory. Moving vectors queried by future tokens into an apparently “historical” prefix does not make them safe. Similarly, masking attention cannot repair a memory read that already includes a future write. A faithful implementation must spell out segment retrieval, token availability and prefix masks, then verify that changing a future token leaves earlier outputs unchanged. Treat the paper's segment-level notation as a design description, not a complete indexing specification."}</Prose>

<TitansProjectionFigure/>

<H3>{"Walk a small gated block all the way through"}</H3>

<Prose>{"Our executable example specializes the MAG topology so every number is inspectable. Inputs are two-vectors. Query, key and value projections are identities. Local attention includes the current token and one previous token, plus an optional persistent vector "}<InlineMath>{"p=(1,0)"}</InlineMath>{". The fast memory is a two-by-two linear matrix, initially zero, with rate 0.5, momentum retention 0.5 and no decay. It writes the observed association "}<InlineMath>{"x_t\\to x_t"}</InlineMath>{", then reads at query "}<InlineMath>{"x_t"}</InlineMath>{". Finally,"}</Prose>

<div className="neural-equation"><MathBlock>{"o_t=a_t\\odot\\tanh(m_t),"}</MathBlock></div>

<Prose>{"where "}<InlineMath>{"a_t"}</InlineMath>{" is the attention result, "}<InlineMath>{"m_t"}</InlineMath>{" the memory read, and "}<InlineMath>{"\\odot"}</InlineMath>{" multiplies corresponding coordinates. This gate can attenuate or reverse a coordinate; it is not a convex probability mixture. These are declared teaching choices, not claimed trained-model defaults. We omit residuals, learned projections, convolution and additional normalization here so the three paths remain visible."}</Prose>

<Prose>{"At the first token "}<InlineMath>{"(1,0)"}</InlineMath>{", the prefix and current token are identical. Attention returns "}<InlineMath>{"(1,0)"}</InlineMath>{". The write sets the first diagonal memory weight to 0.5, so the read is "}<InlineMath>{"(0.5,0)"}</InlineMath>{". Output: "}<InlineMath>{"(\\tanh(0.5),0)\\approx(0.462117,0)"}</InlineMath>{"."}</Prose>

<Prose>{"At the second token "}<InlineMath>{"(0,1)"}</InlineMath>{", attention sees "}<InlineMath>{"(1,0),(1,0),(0,1)"}</InlineMath>{". Dot each row with the query and divide by the square root of its dimension, giving scores "}<InlineMath>{"(0,0,1/\\sqrt2)"}</InlineMath>{". Softmax exponentiates the scores and divides by their sum: the resulting weights are approximately "}<InlineMath>{"(0.248255,0.248255,0.503490)"}</InlineMath>{". Their weighted sum of the three rows gives "}<InlineMath>{"a=(0.496510,0.503490)"}</InlineMath>{". The memory becomes"}</Prose>

<div className="neural-equation"><MathBlock>{"W=\\begin{bmatrix}0.75&0\\\\0&0.5\\end{bmatrix}."}</MathBlock></div>

<Prose>{"The first diagonal changed from 0.5 to 0.75 because of retained momentum, even though the new key points along the second axis. The new memory read is "}<InlineMath>{"(0,0.5)"}</InlineMath>{", and the output is approximately "}<InlineMath>{"(0,0.232671)"}</InlineMath>{"."}</Prose>

<Prose>{"Remove the prefix, replaying from the same initial state. At the second token the memory is unchanged, but attention has only two rows. Its second coordinate is now 0.669762, and the gated output's second coordinate becomes 0.309508. This intervention reveals the prefix's route through attention. At the first token, removing that duplicate prefix leaves the output unchanged: a useful control case."}</Prose>

<TitansGatedLab/>

<H2>{"6. Two learning loops, with two different jobs"}</H2>

<Prose>{""}<strong>{"Deeper branch."}</strong>{" The inner loop writes associations into fast weights for the current sequence. The outer loop learns the projections, gates, initial conditions and other model parameters that make those writes useful for the final task. A language-model outer objective can still be next-token cross-entropy. The inner target is a learned representation derived from available input, not the withheld answer token that the model is supposed to predict."}</Prose>

<TitansOuterFigure/>

<Prose>{"You can understand this with one scalar. Let the initial weight be 0. Write key 2 → value 1 with a learnable rate "}<InlineMath>{"\\theta"}</InlineMath>{". The inner gradient is −2, so "}<InlineMath>{"w'=2\\theta"}</InlineMath>{". Query at 3 and compare to outer target 2:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\hat y=6\\theta,\\qquad L_{\\text{outer}}=\\tfrac12(6\\theta-2)^2,\n\\qquad \\frac{dL_{\\text{outer}}}{d\\theta}=6(6\\theta-2)."}</MathBlock></div>

<Prose>{"At "}<InlineMath>{"\\theta=0.25"}</InlineMath>{", the new weight is 0.5, prediction 1.5, outer loss 0.125 and outer derivative −3. Gradient descent on the rate would increase it locally, making this subsequent query more accurate. The outer objective has taught something about how to write."}</Prose>

<Prose>{"For a general parameter "}<InlineMath>{"\\psi"}</InlineMath>{", differentiation through "}<InlineMath>{"\\phi'=\\phi-\\theta\\nabla_\\phi\\ell(\\phi;\\psi)"}</InlineMath>{" involves how the inner gradient changes with "}<InlineMath>{"\\psi"}</InlineMath>{". With many steps it also includes how previous fast weights affect later writes. Automatic differentiation can evaluate these paths without materializing a dense Hessian. The "}<a href={"https://arxiv.org/html/2407.04620v1#S2.SS2"}>{"TTT paper's inner/outer discussion"}</a>{" is a useful companion."}</Prose>

<Prose>{"In PyTorch, "}<code>{"autograd.grad(..., create_graph=True)"}</code>{" retains a graph for differentiating the computed gradient. For the nonlinear example in § 4, use outer target −0.3 at its query and half-squared outer loss. The resulting outer rate derivative agrees with a centered finite difference: −0.00249327930752 versus −0.00249327930749 at rate 0.1. That is the same differentiation idea as the hand scalar calculation, now through a SiLU network with a different query."}</Prose>

<TitansOuterLab/>

<Prose>{"At inference we only need the current inner write, not an outer derivative through the entire session. The memory cell's ordinary mode detaches the new weights and momentum after each write, then enables gradients on the new weights for the next local write. Detachment at this point bounds retained autograd history. Using that mode during purported full outer training would remove paths from the objective and change what is being optimized."}</Prose>

<Prose>{""}<code>{"eval()"}</code>{" concerns module behavior such as dropout; it is separate from disabling automatic differentiation. A surrounding "}<code>{"no_grad()"}</code>{" would prevent the inner loss from recording the reverse-mode computation required for a write. Keep gradient recording active for that computation, even while avoiding a session-long outer graph. The "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html"}>{"PyTorch gradient API"}</a>{" and "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html"}>{"no_grad semantics"}</a>{" document the distinction."}</Prose>

<H2>{"7. A real stream: forecasting daily bike rentals"}</H2>

<Prose>{"The logbook question becomes concrete with "}<a href={"https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"}>{"UCI's Bike Sharing dataset"}</a>{": 731 daily totals from Washington, DC's Capital Bikeshare system in 2011–2012. Hadi Fanaee-T supplied this dataset for studying rentals and their relation to conditions and events. We use the daily observations, not the larger hourly file. The retained "}<a href={"/learn-assets/titans-multi-memory-architecture/bike-sharing-daily.csv"}>{"CSV"}</a>{", "}<a href={"/learn-assets/titans-multi-memory-architecture/source-description.txt"}>{"provider description"}</a>{" and "}<a href={"/learn-assets/titans-multi-memory-architecture/data-provenance.md"}>{"provenance"}</a>{" make the exercise usable offline; UCI licenses the data under CC BY 4.0."}</Prose>

<Prose>{"Our question is: "}<strong>{"after training a small prediction rule on 2011, does allowing it to update after each newly observed day improve its 2012 forecasts?"}</strong>{" This isolates online neural adaptation on a real stream. The experiment uses observed count targets after they arrive, rather than Titans' learned latent self-supervised targets. It trains an ordinary initial predictor rather than a complete end-to-end Titans language model. The three-branch computation and outer-learning mechanism were exposed separately above so you can identify exactly which part this study exercises."}</Prose>

<H3>{"Specify when a number becomes available"}</H3>

<Prose>{"Assume a day's total is available at that day's end. Before tomorrow arrives, use the preceding seven totals and the weekday of the day being forecast. No future weather observations or components of tomorrow's total enter the input."}</Prose>

<TitansTimelineFigure/>

<Prose>{"The count mean and population standard deviation are fitted on 2011 only: approximately 3405.762 and 1376.864. Standardize counts using those constants. A key contains seven past standardized counts, followed by sine and cosine of the target weekday; normalize the resulting nine-vector to unit length. The memory has shape 9→8→1 with SiLU and biases. Its output is a standardized count, transformed back to rentals for scoring. This preprocessing is a declared compact teaching representation, not a claim that it is an optimal forecasting feature set."}</Prose>

<Prose>{"Fit 358 examples from 2011: the first seven days supply history and targets begin on day 8. Use full-batch Adam with rate 0.01 for 1000 updates. Repeat seeds 3, 7, 19 to show initialization variation. Make two copies of each fitted model: one stays frozen, while the other uses rate 0.005, momentum retention 0.5 and decay 0.0001 after each observed 2012 target. Those choices, the seeds and the simple baselines were fixed in the "}<a href={"/learn-assets/titans-multi-memory-architecture/experiment-protocol.md"}>{"experiment protocol"}</a>{" before running this comparison."}</Prose>

<Prose>{"Record a forecast first. Observe the day's target second. Update the adaptive model third. Report the first 183 forecasts as a development window and the following 183 as assessment. No settings were selected from either report, and the adaptive state carries forward across their boundary. The first window ends on 1 July 2012 because 2012 is a leap year; the assessment begins on 2 July."}</Prose>

<H3>{"One actual forecast and write"}</H3>

<Prose>{"For seed 3, both copies forecast 2609.616 rentals for 1 January 2012. They agree because adaptation has not yet made a write. The observed total is 2294, so the count error is 315.616. Dividing by the fitted scale gives the standardized residual; its half-square is 0.02627285. The gradient norm is 0.67017273 and the parameter-change norm is 0.00352720. Those last two are measured in parameter space, not rentals."}</Prose>

<Prose>{"For 2 July 2012, the frozen copy forecasts 4707.926 and the adaptive copy 5944.152. The actual count is 6227. The adaptive model has incorporated earlier 2012 outcomes. This is the same state-carrying principle as the continued gated sequence in § 5, now applied to dated observations and measured forecast error."}</Prose>

<H3>{"Run and inspect the full experiment"}</H3>

<Prose>{"Save the supplied files together. Python, NumPy and PyTorch are required. The recorded run used Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu, float64 with one CPU thread. These are tested versions, not a requirement to install the latest release."}</Prose>

<CodeBlock language={"sh"}>{"python -B memory_mechanisms.py\npython -B rental_memory_study.py\npython -B check_author_packet.py"}</CodeBlock>

<Prose>{"The complete "}<a href={"/learn-assets/titans-multi-memory-architecture/rental_memory_study.py"}>{"study program"}</a>{" includes data loading, feature construction, fitting, replay, scoring and result export. Its central replay operation follows the three moments above:"}</Prose>

<CodeBlock language={"python"}>{"prediction = float(read_memory(weights, key).detach())\nnext_weights, next_momentum, loss, gradients = write_memory(\n    weights, momentum, key, observed_value,\n    rate=0.005, retention=0.5, decay=0.0001,\n)\nweights, momentum = next_weights, next_momentum"}</CodeBlock>

<Prose>{"Here "}<code>{"observed_value"}</code>{" is supplied only after saving "}<code>{"prediction"}</code>{". The complete program retains every date, forecast, residual, gradient norm and update norm in "}<a href={"/learn-assets/titans-multi-memory-architecture/rental-results.json"}>{"rental-results.json"}</a>{", along with initial/final weights and environment identity. The short excerpt shows the timing; use the full linked program to run it."}</Prose>

<TitansProgram file="rental_memory_study.py"/>

<Prose>{"Mean absolute error, or "}<strong>{"MAE"}</strong>{", averages the absolute difference between forecast and observation. Its units here are rentals per day. "}<strong>{"RMSE"}</strong>{" is the square root of the mean squared error; squaring gives large misses more influence. The complete result table retains both. For a readable first comparison, here is MAE, rounded to two decimals:"}</Prose>

<NeuralTable caption={"Run and inspect the full experiment"} headers={[<>{"Procedure"}</>,<>{"Development: 1 Jan–1 Jul"}</>,<>{"Assessment: 2 Jul–31 Dec"}</>]} rows={[[<>{"Previous day's count"}</>,<>{"896.54"}</>,<>{"843.81"}</>],[<>{"Count seven days earlier"}</>,<>{"1092.26"}</>,<>{"1128.53"}</>],[<>{"Frozen network, seed 3"}</>,<>{"1334.04"}</>,<>{"1767.05"}</>],[<>{"Adaptive network, seed 3"}</>,<>{"925.34"}</>,<>{"823.54"}</>],[<>{"Frozen network, seed 7"}</>,<>{"1342.21"}</>,<>{"1781.81"}</>],[<>{"Adaptive network, seed 7"}</>,<>{"1007.78"}</>,<>{"867.34"}</>],[<>{"Frozen network, seed 19"}</>,<>{"1392.46"}</>,<>{"1813.87"}</>],[<>{"Adaptive network, seed 19"}</>,<>{"875.92"}</>,<>{"805.50"}</>]]} />

<Prose>{"All three adaptive copies improve over their matched frozen networks. Yet the simple previous-day baseline beats two adaptive runs in development and one in assessment. The practical conclusion is to keep that baseline in the comparison. The extra computation provides an observed benefit over freezing this predictor, but initialization and the comparator affect whether the added complexity is useful."}</Prose>

<TitansPairedFigure/><TitansRentalLab/>

<Prose>{"Try inspecting a date with a large error and its following write. A high error could reflect a poor representation, a real change, or an unusual event. The count alone cannot tell you which explanation caused it. State an additional measurement you would need before declaring that the model has detected a particular event."}</Prose>

<H3>{"A one-line timing error can manufacture perfect performance"}</H3>

<Prose>{"For a constructed scalar memory, let "}<InlineMath>{"w=0"}</InlineMath>{", key 1, observed target 4 and rate 1, with no momentum or decay. The forecast before observing the target is 0, an error of −4. After writing, the weight becomes 4. Evaluating against the same target at that point gives zero error. That measures fitting an already revealed target, not forecasting it."}</Prose>

<Prose>{"The author checks perturb a later daily count and confirm that earlier forecasts stay unchanged. The first affected forecast comes after the changed count becomes available. This is a direct check of the dependency rule, rather than trusting a label such as “causal” or “test set.”"}</Prose>

<H3>{"Build the write rule, then choose what stays differentiable"}</H3>

<Prose>{""}<a href={"/learn-assets/titans-multi-memory-architecture/memory_mechanisms.py"}>{"memory_mechanisms.py"}</a>{" owns the explicit linear residual/outer-product gradient, momentum and decay in "}<code>{"linear_write"}</code>{", plus the small complete gated memory/attention composition in "}<code>{"gated_sequence"}</code>{". "}<a href={"/learn-assets/titans-multi-memory-architecture/neural_memory.py"}>{"neural_memory.py"}</a>{" is the ordinary research implementation for a nonlinear memory: "}<code>{"read_memory"}</code>{" evaluates its two-layer network from explicit parameter tensors and "}<code>{"write_memory"}</code>{" obtains the write-loss derivatives with "}<code>{"torch.autograd.grad"}</code>{". It returns new parameter and momentum tuples rather than silently mutating a globally shared model. Autograd is the reused derivative engine; the new mechanism is the loss-driven persistent memory update."}</Prose>

<TitansProgram file="memory_mechanisms.py"/>

<Prose>{"The "}<code>{"differentiable"}</code>{" switch is a learning decision. With "}<code>{"True"}</code>{", "}<code>{"create_graph=True"}</code>{" retains the derivative graph through a write so an outer objective can learn a write rate or initializer. With "}<code>{"False"}</code>{", the newly returned state is detached and made a new leaf for the next write; this bounds the retained history during ordinary online replay but removes earlier-write meta-gradients. "}<code>{"rental_memory_study.py::replay"}</code>{" chooses that causal online path, preserving the forecast-before-observation timeline. Both paths implement complete writes; they optimize different derivative contracts."}</Prose>

<Prose>{"No one-call Titans package is necessary for this route. Standard tensors, functional layer evaluation, autograd and the explicit state tuple are ordinary tools for research on changing fast weights. The linear update's outer product costs O(d_key d_value), and a nonlinear write costs a memory-network forward/backward plus parameter-sized momentum/state. Differentiating through many writes also retains their graphs; constant-sized *carried values* do not imply constant training-memory cost."}</Prose>

<Prose>{""}<strong>{"Take control."}</strong>{" In "}<code>{"memory_mechanisms.py::nonlinear_outer"}</code>{", compare the derivative of the outer loss with respect to the write rate using central differences and "}<code>{"write_memory(..., differentiable=True)"}</code>{". Then detach the new state and inspect which derivative disappears. Keep the initial weights, key, target and query fixed."}</Prose>

<details><summary>Hint and reasoned solution</summary>

<Prose>{"Central differences must rebuild the same initial state for rates η+ε and η−ε. The differentiable path includes how changing η changes the written weights and the later query output. Detaching makes that written value an independent leaf, so the outer loss has no graph path to η through the write; asking for it may return unused/None or raise unless unused inputs are explicitly permitted. That does not mean the numerical function is insensitive to η. It means the chosen differentiation contract discarded that dependence. Restore "}<code>{"differentiable=True"}</code>{" for the meta-learning question, and use the detached mode for the intentionally bounded replay question. Compare a smaller ε to diagnose cancellation rather than assuming the smallest possible ε is best."}</Prose>

</details>

<H2>{"8. Parallelize a declared update rule"}</H2>

<Prose>{""}<strong>{"Deeper branch."}</strong>{" Each online step can depend on the previous fast weights twice: directly in the weight recurrence and inside the gradient calculation. The second dependency is expensive for a nonlinear memory. Calculating a chunk's gradients at one shared starting state removes that dependency inside the chunk. Prefix sums or scans can then combine the precomputed updates."}</Prose>

<Prose>{"This changes the update convention. Take a scalar memory "}<InlineMath>{"M_w(1)=w"}</InlineMath>{", initial weight 0, target sequence 1 then 2, rate 0.5, and no momentum or decay:"}</Prose>

<NeuralTable caption={"8. Parallelize a declared update rule"} headers={[<>{"Method"}</>,<>{"First gradient → weight"}</>,<>{"Second gradient → weight"}</>]} rows={[[<>{"Recompute at the current weights"}</>,<>{"−1 → 0.5"}</>,<>{""}<InlineMath>{"0.5-2=-1.5"}</InlineMath>{" → 1.25"}</>],[<>{"Both gradients at the chunk anchor 0"}</>,<>{"−1 → 0.5"}</>,<>{""}<InlineMath>{"0-2=-2"}</InlineMath>{" → 1.5"}</>]]} />

<Prose>{"The difference is where the second gradient is evaluated. Both calculations use a running weight state; one uses a stale anchor when determining the descent direction. Chunk size 1 recovers the sequential rule, and a zero rate makes both leave the weight unchanged."}</Prose>

<TitansChunkLab/>

<Prose>{"With fixed gradient inputs "}<InlineMath>{"u_t"}</InlineMath>{", momentum obeys an affine recurrence "}<InlineMath>{"S_t=\\eta_t S_{t-1}-\\theta_t u_t"}</InlineMath>{". Two transformations "}<InlineMath>{"F_1(s)=a_1s+b_1"}</InlineMath>{" and "}<InlineMath>{"F_2(s)=a_2s+b_2"}</InlineMath>{" compose as"}</Prose>

<TitansScanFigure/>

<div className="neural-equation"><MathBlock>{"F_2(F_1(s))=(a_2a_1)s+(a_2b_1+b_2)."}</MathBlock></div>

<Prose>{"Function composition is associative, which permits a scan organized as a tree rather than a strictly serial chain. That identity explains the parallel opportunity. It does not make the original nonlinear gradient evaluations independent. The "}<a href={"https://arxiv.org/html/2501.00663v1#S3.SS2"}>{"Titans parallelization section"}</a>{" and "}<a href={"https://arxiv.org/html/2407.04620v1#S2.SS4"}>{"TTT mini-batch derivation"}</a>{" discuss this distinction."}</Prose>

<Prose>{"Likewise, accumulating gradients over examples at fixed ordinary model weights is not the same operation as repeatedly changing those weights between examples. The next lesson makes that distinction operational in a conventional training loop, including uneven batch sizes and the last partial update."}</Prose>

<H2>{"9. Count the state you actually retain"}</H2>

<Prose>{""}<strong>{"Deeper branch."}</strong>{" Separate shared model parameters, state for each active sequence, temporary workspaces and the graph retained for outer training. Their scaling can differ."}</Prose>

<Prose>{"Let a bias-free memory have dimensions "}<InlineMath>{"d\\to H\\to d"}</InlineMath>{". It contains "}<InlineMath>{"2dH"}</InlineMath>{" fast weights. If both weights and momentum use four-byte float32, their combined storage is "}<InlineMath>{"2(2dH)4"}</InlineMath>{" bytes per memory instance. With "}<InlineMath>{"d=2048"}</InlineMath>{" and "}<InlineMath>{"H=512"}</InlineMath>{", that is 16 MiB per layer. With 24 such layers and 4 independent sequences, it becomes 1.5 GiB."}</Prose>

<Prose>{"For an attention cache with batch "}<InlineMath>{"B"}</InlineMath>{", attention layers "}<InlineMath>{"L_a"}</InlineMath>{", retained positions "}<InlineMath>{"T_r"}</InlineMath>{", KV heads "}<InlineMath>{"h_{kv}"}</InlineMath>{", head dimension "}<InlineMath>{"d_h"}</InlineMath>{" and "}<InlineMath>{"b"}</InlineMath>{" bytes per value,"}</Prose>

<div className="neural-equation"><MathBlock>{"\\text{KV bytes}=2B L_a T_r h_{kv}d_h b."}</MathBlock></div>

<Prose>{"The leading 2 accounts for keys and values. Use the number of KV heads, not automatically the number of query heads. A 2048-position window with batch 4, 24 attention layers, 8 KV heads, head dimension 128 and two-byte elements uses 0.75 GiB. Retaining 131072 positions under the same assumptions instead uses 48 GiB. These are exact payload calculations for a hypothetical configuration, not the memory footprint of a named released model."}</Prose>

<TitansPayloadFigure/>

<Prose>{"For fixed dimensions and a fixed window, a forward read and local write cost a bounded amount per position; total work grows with sequence length. A two-layer memory read has work proportional to "}<InlineMath>{"dH"}</InlineMath>{", and its gradient adds work on the same dimensions. Local attention adds a window-dependent term. Projection, feedforward, normalization and other architecture costs still exist. An outer training pass can retain activations or reconstruct them, so its peak memory does not follow solely from the inference-state formula."}</Prose>

<Prose>{"Wall-clock latency also depends on chunking, tensor layout, batch size, precision, kernels, transfers and synchronization. Multiplying FLOPs by a peak hardware specification is not a measured runtime. Benchmark the actual causal update convention, context distribution and serving workload, including quality at the required retrieval distances."}</Prose>

<H2>{"10. Read the research without turning it into a promise"}</H2>

<Prose>{""}<strong>{"Deeper branch."}</strong>{" The canonical source is "}<a href={"https://arxiv.org/html/2501.00663v1"}>{"Behrouz, Zhong and Mirrokni, *Titans: Learning to Memorize at Test Time*"}</a>{". Its arXiv submission is 31 December 2024; Google's publication entry labels it 2025. The paper studies memory design, integration, language modeling, retrieval/reasoning, forecasting, DNA modeling, efficiency and ablations. It is useful to read those as separate questions."}</Prose>

<Prose>{"For example, the paper's Table 2 reports single-needle retrieval at 2K, 4K, 8K and 16K. Its longer-context BABILong results concern a different benchmark and include different fine-tuning/few-shot settings. Combining them into a smooth invented “accuracy up to 2M” curve would conceal those distinctions. A model that processes a long input has not thereby demonstrated that every fact in it remains recoverable."}</Prose>

<Prose>{"Depth and component ablations ask whether changes help under the reported setup. They can motivate a new experiment, while the nonlinear-memory proof above explains a representational possibility independently of those scores. The paper also makes a theoretical expressivity claim about state tracking; this lesson does not supply its complexity-theory proof. It should not be recast as a guarantee that a trained model solves every difficult reasoning task."}</Prose>

<Prose>{"The authors' "}<a href={"https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/"}>{"December 2025 Titans/MIRAS overview"}</a>{" introduces a useful research lens: choose a memory structure, its learning objective, its retention rule and its update algorithm. This gives you a way to classify an experiment. Replacing squared loss with a robust loss changes what errors drive writes; changing retention alters what survives; changing the memory network alters the functions it can represent. Those are distinct interventions. The overview is a conceptual companion; the precise equations and experimental conditions should be checked in the underlying papers."}</Prose>

<H3>{"A less obvious application: a changing visual stream"}</H3>

<Prose>{"The TTT research line also explores adaptation for vision. In "}<a href={"https://yueatsprograms.github.io/ttt/home.html"}>{"Sun and colleagues' 2020 project"}</a>{", the visible test input supplies a self-supervised task such as predicting an applied image rotation. Shared features are updated using that auxiliary task before the main prediction, without revealing the main class label. It demonstrates why “learning at test time” need not mean looking at a test answer."}</Prose>

<TitansResearchFigure/>

<Prose>{"Map the roles carefully: the observed image provides the auxiliary target; feature weights adapt; the classification head then predicts the withheld class. For an online stream, carrying state forward additionally assumes that consecutive inputs have a relationship worth exploiting. A sudden camera change could make that assumption less helpful. The natural investigation is to compare reset and carry-forward policies under a declared sequence of shifts, with the same starting model and without using class labels for adaptation."}</Prose>

<Prose>{"That application differs from our supervised rental replay and from a Titans key/value memory. Putting them next to each other helps isolate a general design question: "}<strong>{"what target is available under the task protocol at the moment a model updates, and how does improving that target help the task we care about?"}</strong>{""}</Prose>

<H3>{"Choose an approach by the work it must do"}</H3>

<Prose>{"For exact document provenance, an external retrieval system can retain text, source identifiers and dates. A fast-weight memory instead returns a learned compressed representation. For ordered streaming patterns, an adaptive state can incorporate each observation as it arrives. Recent attention helps address individual accessible positions. These approaches can be combined; “ordered input” is not a reason that retrieval is inherently impossible."}</Prose>

<Prose>{"Before choosing a design, define the question, target availability, necessary evidence, tolerated error, reset boundary, expected sequence lengths and resource budget. Then compare an existing simple baseline with the proposed method. An architecture diagram alone cannot determine which released system satisfies those requirements. This lesson does not infer unpublished product internals or make current checkpoint-availability claims from a research paper."}</Prose>

<H2>{"11. Diagnose the mechanism you actually ran"}</H2>

<NeuralTable caption={"11. Diagnose the mechanism you actually ran"} headers={[<>{"Observation"}</>,<>{"Useful next comparison"}</>,<>{"What it can distinguish"}</>]} rows={[[<>{"High loss, tiny gradient"}</>,<>{"Inspect keys and parameter sensitivities; try the zero-key control"}</>,<>{"A mismatch can be large in output space yet hard to affect through the chosen parameters"}</>],[<>{"Weights change despite zero new gradient"}</>,<>{"Log old/new momentum and decay separately"}</>,<>{"Carrying an earlier update versus a new gradient-driven write"}</>],[<>{"Inner loss improves, task score worsens"}</>,<>{"Hold the sequence fixed and compare query/task outputs before and after writes"}</>,<>{"Fitting associations versus improving the outer task"}</>],[<>{"Reading in chunks changes outputs"}</>,<>{"Compare complete state carry and the gradient-anchor convention"}</>,<>{"Lost state versus a different algorithm"}</>],[<>{"A new request depends on an earlier one"}</>,<>{"Compare isolated initial states with deliberately shared state"}</>,<>{"An unintended cross-request dependency"}</>],[<>{"Larger magnitudes or nonfinite values"}</>,<>{"Inspect the first offending input, residual, gradient and update"}</>,<>{"Scale, update, data or arithmetic problems that deserve different repairs"}</>],[<>{"A forecast becomes perfect after adaptation"}</>,<>{"Check when the target first entered the computation"}</>,<>{"Forecasting versus scoring a fitted target"}</>]]} />

<Prose>{"Do not jump from a symptom to a unique cause. For example, reducing a rate may prevent an immediate numerical failure while leaving a target-leakage bug intact. A useful debugging record names the hypothesis, one controlled change, what was held fixed and the observation that would distinguish explanations."}</Prose>

<section className="lesson-ending lesson-ending--practice" data-lesson-ending="practice"><H2>{"12. Practice: carry the idea to a changed case"}</H2>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"1. An old association moves"}</H3>

<Prose>{"Start from "}<InlineMath>{"w=(2,-1)"}</InlineMath>{", write key "}<InlineMath>{"(1,2)"}</InlineMath>{" → value 3 using half-squared loss, rate 0.1 and no momentum/decay. Compute the new weights and the change in the answer to query "}<InlineMath>{"(2,1)"}</InlineMath>{". Verify the change once by direct readout and once by the overlap identity."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Compute the prediction at the write key first. The query is a separate vector; use it only when calculating the old and new reads or the overlap."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The write prediction is "}<InlineMath>{"2-2=0"}</InlineMath>{", residual −3 and gradient "}<InlineMath>{"(-3,-6)"}</InlineMath>{". New weights are "}<InlineMath>{"(2.3,-0.4)"}</InlineMath>{". The query read changes from "}<InlineMath>{"4-1=3"}</InlineMath>{" to "}<InlineMath>{"4.6-0.4=4.2"}</InlineMath>{", an increase of 1.2. The overlap is "}<InlineMath>{"(1,2)^\\top(2,1)=4"}</InlineMath>{", so "}<InlineMath>{"-0.1(-3)(4)=1.2"}</InlineMath>{" gives the same change. The key tells the memory what to fit; the query exposes a consequence elsewhere."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"2. Is decay a reset?"}</H3>

<Prose>{"For a scalar weight, let "}<InlineMath>{"w=3"}</InlineMath>{", old momentum 0.4, new gradient 0, momentum retention 0.5 and "}<InlineMath>{"\\alpha=1"}</InlineMath>{". What is the new weight? Specify a true fresh-sequence reset when the chosen initial weight is 0.7."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Calculate momentum before applying the weight recurrence. Distinguish the zero vector from the configured initial state."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"New momentum is 0.2; new weight is "}<InlineMath>{"0\\cdot3+0.2=0.2"}</InlineMath>{". A fresh sequence restores the weight to 0.7, momentum to 0, and clears its local context and other sequence state. No claim about the content being forgotten follows solely from a single gate value."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"3. Change the chunk rule"}</H3>

<Prose>{"Use initial scalar weight 1, keys 1, targets 3 then −1, rate 0.25 and no momentum/decay. Calculate the final weight with current-state gradients and with both gradients at the starting anchor. What happens if the chunk size is 1?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"The first step agrees. For the second gradient, explicitly write the weight at which the residual is evaluated."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"First gradient is −2, giving 1.5. Sequentially the next gradient is "}<InlineMath>{"1.5-(-1)=2.5"}</InlineMath>{", giving "}<InlineMath>{"1.5-0.25(2.5)=0.875"}</InlineMath>{". With anchor 1, the next gradient is 2, giving 1.0. Chunk size 1 refreshes the anchor before each gradient and yields 0.875. Both methods received the same observations; their descent directions differed."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"4. Judge the rental result"}</H3>

<Prose>{"Use the assessment results to answer: did adaptation help the seed 7 network, and would you choose it over the previous-day baseline on MAE alone? Design a new comparison that could justify a more expensive predictor without reusing the same assessment interval for unlimited model selection."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Those are two different comparisons. State the decision criterion before proposing a tuning experiment."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Seed 7's adaptive MAE 867.34 improves greatly over its frozen MAE 1781.81, but remains above the previous-day baseline 843.81. On these assessment MAEs alone, select the simple baseline. A follow-up could predeclare a later dated holdout, choose features/rates on earlier rolling development intervals, and compare the locked procedure's MAE, large-error behavior and compute cost on the later interval. The retained data end in 2012, so a genuinely later real holdout would require additional data or an explicitly redesigned study; it cannot be invented by relabeling the already inspected results."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"5. Learn the write rate"}</H3>

<Prose>{"Start with scalar weight 0. Write key 1 → value 2 with rate "}<InlineMath>{"\\theta"}</InlineMath>{", then query at 2 with outer target 3. Derive the outer loss and its rate derivative. Evaluate them at "}<InlineMath>{"\\theta=0.5"}</InlineMath>{"."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Express the new weight as a function of the rate before substituting the number. Otherwise you can accidentally discard the dependency you need to differentiate."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Inner gradient −2 gives "}<InlineMath>{"w'=2\\theta"}</InlineMath>{"; query output is "}<InlineMath>{"4\\theta"}</InlineMath>{". Outer loss is "}<InlineMath>{"\\tfrac12(4\\theta-3)^2"}</InlineMath>{", derivative "}<InlineMath>{"4(4\\theta-3)"}</InlineMath>{". At 0.5, prediction 2, loss 0.5 and derivative −4. A small increase in the write rate decreases this outer loss locally. A program that detaches the updated weight before computing the outer objective would lose this rate path."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"6. A serving-state calculation"}</H3>

<Prose>{"A hypothetical bias-free memory has dimensions 512→128→512. Its weights and momentum are float32. There are 12 memory layers and 8 independent requests. Calculate their combined payload, then name at least three missing categories before declaring whether the system fits on a device."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Count two weight matrices, a momentum copy, bytes per entry, layers and requests. Divide by "}<InlineMath>{"2^{20}"}</InlineMath>{" for MiB."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Each layer/request has "}<InlineMath>{"2(512)(128)=131072"}</InlineMath>{" weights. Weights plus momentum take "}<InlineMath>{"131072\\times2\\times4=1048576"}</InlineMath>{" bytes, exactly 1 MiB. Across 12 layers and 8 requests this is 96 MiB. Missing categories include shared base-model weights, attention K/V, convolution and position/chunk state, temporary workspace, and training activations if doing an outer training pass. Dtypes can differ between categories, so use the actual ones in a complete calculation."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"7. Explain a delayed reversal"}</H3>

<Prose>{"Momentum is at its steady state −0.5 under constant gradient 1, rate 0.05 and retention 0.9. The gradient becomes −1. After how many steps does momentum first become positive? Why is that different from reaching its new steady state?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"The new steady state is 0.5. Solve the recurrence for the distance from that value."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"After n new steps, "}<InlineMath>{"S_n=0.5-0.9^n"}</InlineMath>{". It becomes positive when "}<InlineMath>{"0.9^n<0.5"}</InlineMath>{", first at n=7. The distance to the new steady state is "}<InlineMath>{"0.9^n"}</InlineMath>{"; it approaches zero asymptotically and requires a declared tolerance for an approximate stopping time. A timescale such as "}<InlineMath>{"1/(1-0.9)=10"}</InlineMath>{" is not the exact sign-reversal answer."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"8. Repair a plausible architecture explanation"}</H3>

<Prose>{"Someone says: “We computed every token's memory read from the initial weights, then wrote the whole sequence afterward. Our attention is causal, so the memory has learned the earlier tokens for every output.” Identify the error and propose a controlled check. Then explain the different error in reading every position from the final post-sequence weights."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Ask which exact memory version produced the read at position t, independently of the attention mask."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"All initial-state reads miss the adaptation from earlier tokens in this call. Reads must use the state prescribed by the declared online or chunk update rule. Compare a whole-call run with a continuation run that carries identical state across a boundary; they should agree when the convention agrees. Using the final post-sequence weights for earlier positions creates the opposite problem: those states can contain future writes. Perturb a future input and check earlier outputs. The attention mask cannot repair either memory-path error."}</Prose>

</details></div></section>

<section className="lesson-ending lesson-ending--next" data-lesson-ending="next"><H2>{"Ready to move on?"}</H2>

<Prose>{"Explain where the three memories store information; perform a write and predict a changed query; distinguish loss, gradient and actual update; identify the inner and outer targets; and justify when a target becomes available. Then reproduce one controlled state comparison and interpret a real result against its simple baseline. These are stronger signs of understanding than remembering the three architecture abbreviations."}</Prose>

<Prose>{"Continue to "}<a href={"/learn/path/full-curriculum/mini-batches-training-loops-gradient-accumulation?module=deep-learning-fundamentals"}>{"Mini-Batches, Training Loops & Gradient Accumulation"}</a>{". It follows this topic in the module and turns the distinction between computing gradients and applying updates into a complete practical loop. The following diagnostics lesson then teaches how to test that loop's behavior."}</Prose></section>

<section className="lesson-ending lesson-ending--resources" data-lesson-ending="resources"><H2>{"References and another way to learn it"}</H2>

<ul><li>{""}<a href={"https://arxiv.org/html/2501.00663v1"}>{"Behrouz, Zhong and Mirrokni: Titans"}</a>{". Primary paper; begin with §§ 3.1 and 4 after the hand example, then read § 3.2 and Appendix C for update conventions. The actual agenda and those sections were read, together with the experiment sections. Mathematical notation sometimes compresses implementation details; use the timing distinctions in this lesson when translating it to code."}</li><li>{""}<a href={"https://arxiv.org/html/2407.04620v1"}>{"Sun and colleagues: Learning to (Learn at Test Time)"}</a>{". Primary paper with a helpful alternative inner/outer-loop explanation and mini-batch derivation. Read §§ 2.1–2.4 after § 6 here. The relevant mechanism, training-view and chunk sections were inspected; this packet does not claim to reproduce the language-model experiments."}</li><li>{""}<a href={"https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/"}>{"Google Research: Titans + MIRAS"}</a>{". An illustrated article by the researchers, dated 4 December 2025, for revisiting the high-level design choices. Its conceptual sections were reviewed. Read the precise paper equations for the update rather than interpreting every everyday “surprise” example literally."}</li><li>{""}<a href={"https://yueatsprograms.github.io/ttt/home.html"}>{"Sun and colleagues' 2020 TTT project and companion explanation"}</a>{", with its "}<a href={"https://www.youtube.com/watch?v=NbuWxmMco30"}>{"ICML talk"}</a>{". A video alternative about test-time self-supervision and distribution shifts, not a Titans tutorial. The project's substantive introduction/method and official talk link were inspected; the video was not watched, and no timestamps are asserted. Useful after the target-availability discussion in § 7."}</li><li>{""}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html"}>{"PyTorch autograd.grad"}</a>{" and "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html"}>{"no_grad"}</a>{". Versioned API references for implementing the two graph-lifetime modes in § 6. Relevant function semantics were read; the supplied programs were actually executed in the environment recorded with their outputs."}</li><li>{""}<a href={"https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"}>{"UCI Bike Sharing"}</a>{", Hadi Fanaee-T, 2013, "}<a href={"https://doi.org/10.24432/C5W894"}>{"DOI 10.24432/C5W894"}</a>{". Dataset and descriptive context for the real application. Use the supplied daily CSV and provenance to reproduce this lesson's exact data input."}</li></ul>

<Prose>{"The original research and author calculations were checked on 13 September 2026. The supplied programs retain the complete experiment and its recorded results; the website investigations expose the same update equations and saved-data replay."}</Prose>
<TitansProgram file="check_author_packet.py"/><TitansDownloads/></section>
</div>};
