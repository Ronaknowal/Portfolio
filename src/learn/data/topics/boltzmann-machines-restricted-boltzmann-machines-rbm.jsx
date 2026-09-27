// Generated from the complete prepared manuscript by scripts/generate-rbm-lesson.mjs.
import { Prose,H2,H3,CodeBlock } from '../../components/content';
import {Math as InlineMath,MathBlock} from '../../components/content/Math.jsx';
import {NeuralTable} from '../../components/lesson-labs/NeuralLessonElements.jsx';
import {RbmEnergyLab,RbmGradientLab,RbmChainLab,RbmReconstructionLab,RbmDigitLab,RbmProgram,RbmGeneralFigure,RbmEnumerationFigure,RbmPersistenceLab,RbmStudyFigure,RbmSampleGallery,RbmFamilyFigure,RbmAisFigure,RbmLibraryLab} from '../../components/lesson-labs/RbmLabs.jsx';
export default { title:'Boltzmann Machines & Restricted Boltzmann Machines (RBM)',readTime:'~80 min read + experiments and practice',content:()=> <div className="neural-lesson neural-lesson-neutral rbm-lesson">
<Prose opening="exploration">{""}<strong>{"Explore as you read."}</strong>{" Edit small-model biases/interactions, data counts, transition/sampling settings and supported retained digit states. Show normalized joint/marginal probabilities, data-model statistics, exact transition mass and sampled chain trajectories simultaneously. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to distinguish energy from normalized likelihood, reconstruction from probability and finite mixing behavior from an equilibrium claim."}</Prose>

<Prose>{"Imagine learning what plausible handwritten digits look like without being told which digit each image represents. A model could assign a score to every possible image, then make images with better scores more probable. It could also use the visible half of an image to reason about the missing half."}</Prose>

<Prose>{"A "}<strong>{"Boltzmann machine"}</strong>{" does this with interacting random variables. A "}<strong>{"restricted Boltzmann machine"}</strong>{", or RBM, removes particular connections so that some otherwise difficult calculations become simple. We will build a model with only three binary switches, calculate every probability, and then train a small model on real digit images. The small model is deliberately chosen so that we can check its approximate training methods against exact answers."}</Prose>

<Prose>{"The previous "}<a href={"/learn/path/full-curriculum/graph-transformers-geometric-deep-learning?module=deep-learning-fundamentals"}>{"Graph Transformers & Geometric Deep Learning lesson"}</a>{" studied how architecture encodes relationships and symmetries. Here a graph has another job: it describes interactions in a probability distribution. A connecting line is not an attention weight or a causal claim."}</Prose>

<Prose opening="route">{""}<strong>{"First pass:"}</strong>{" sections 1–5 establish the mechanism, sections 6–7 carry out and interpret the experiment, and practice 1–5 checks your understanding. Section 8 and the later problems develop deeper connections. You need weighted sums, elementary probability and the idea of a gradient. We introduce the needed conditional probabilities, expectations and normalizing constants locally; no statistical physics background is assumed."}</Prose>

<H2>{"1. From interacting switches to a probability distribution"}</H2>

<Prose>{"A binary variable can be zero or one. For an image, it might represent whether a particular pixel is dark enough to count as ink. A configuration is the full list of those zeros and ones. We want some configurations to be common, others rare, while leaving room for uncertainty."}</Prose>

<Prose>{"Assign a real number "}<InlineMath>{"E(s)"}</InlineMath>{" to a configuration "}<InlineMath>{"s"}</InlineMath>{". We call this its "}<strong>{"energy"}</strong>{", with lower energy meaning greater preference. In this lesson energy is a dimensionless model score, not a measured number of joules. Turn it into a probability using"}</Prose>

<div className="neural-equation"><MathBlock>{"p(s)=\\frac{e^{-E(s)}}{Z},\\qquad Z=\\sum_{s'}e^{-E(s')}."}</MathBlock></div>

<Prose>{"The sum visits every allowed configuration. "}<InlineMath>{"Z"}</InlineMath>{", the "}<strong>{"partition function"}</strong>{", makes all probabilities add to one. If two energies are 0 and "}<InlineMath>{"-\\log 3"}</InlineMath>{", their unnormalized weights are 1 and 3, so their probabilities are "}<InlineMath>{"1/4"}</InlineMath>{" and "}<InlineMath>{"3/4"}</InlineMath>{". An energy difference becomes a probability ratio."}</Prose>

<Prose>{"Adding 100 to every energy multiplies every weight by the same "}<InlineMath>{"e^{-100}"}</InlineMath>{". It changes "}<InlineMath>{"Z"}</InlineMath>{" by that factor and leaves all probabilities unchanged. Thus an isolated energy value cannot establish how likely something is. Changing one state's energy relative to the others can."}</Prose>

<Prose>{"For a general binary Boltzmann machine, one possible parameterization is"}</Prose>

<div className="neural-equation"><MathBlock>{"E(s)=-\\sum_i c_i s_i-\\sum_{i<j}J_{ij}s_is_j."}</MathBlock></div>

<Prose>{"The bias "}<InlineMath>{"c_i"}</InlineMath>{" favors switch "}<InlineMath>{"i"}</InlineMath>{" being on. A positive interaction "}<InlineMath>{"J_{ij}"}</InlineMath>{" favors the two switches being on together; a negative one discourages that joint event. Connections are undirected and counted once. A general Boltzmann machine may have sparse connections; it does not have to connect every pair."}</Prose>

<Prose>{"For two visible switches with zero biases and "}<InlineMath>{"J_{12}=\\log3"}</InlineMath>{", the four configurations "}<InlineMath>{"00,01,10,11"}</InlineMath>{" have unnormalized weights "}<InlineMath>{"1,1,1,3"}</InlineMath>{". Therefore "}<InlineMath>{"p(11)=1/2"}</InlineMath>{", while each marginal on-probability is "}<InlineMath>{"2/3"}</InlineMath>{". Independence would predict "}<InlineMath>{"4/9"}</InlineMath>{", which is different. "}<strong>{"A Boltzmann machine can model dependence even without hidden variables"}</strong>{" when visible-to-visible interactions are present."}</Prose>

<Prose>{"We will instead use hidden variables to create useful dependencies while imposing a simpler graph."}</Prose>

<RbmGeneralFigure />

<H2>{"2. What the restriction buys us"}</H2>

<Prose>{"An RBM has observed "}<strong>{"visible"}</strong>{" variables "}<InlineMath>{"v_1,\\ldots,v_D"}</InlineMath>{" and unobserved "}<strong>{"hidden"}</strong>{" variables "}<InlineMath>{"h_1,\\ldots,h_H"}</InlineMath>{". Hidden variables can combine evidence from multiple pixels. They are learned latent features, not automatically digit labels or human-interpretable concepts."}</Prose>

<Prose>{"Connections run between the two groups. There are no visible-to-visible or hidden-to-hidden connections. This is a "}<strong>{"bipartite"}</strong>{" graph. Our Bernoulli–Bernoulli RBM has binary variables on both sides and energy"}</Prose>

<div className="neural-equation"><MathBlock>{"E(v,h)=-a^Tv-b^Th-v^TWh."}</MathBlock></div>

<Prose>{""}<InlineMath>{"W"}</InlineMath>{" is "}<InlineMath>{"D\\times H"}</InlineMath>{", "}<InlineMath>{"a"}</InlineMath>{" contains "}<InlineMath>{"D"}</InlineMath>{" visible biases, and "}<InlineMath>{"b"}</InlineMath>{" contains "}<InlineMath>{"H"}</InlineMath>{" hidden biases. These shapes are also the storage convention in the complete program. Some libraries transpose "}<InlineMath>{"W"}</InlineMath>{"; the equation, not a variable name, determines which axis means what."}</Prose>

<RbmEnergyLab />

<H3>{"Condition on one layer; the other separates"}</H3>

<Prose>{"Fix the visible switches. Everything involving a particular hidden switch "}<InlineMath>{"h_j"}</InlineMath>{" becomes"}</Prose>

<div className="neural-equation"><MathBlock>{"-h_j\\left(b_j+\\sum_i v_iW_{ij}\\right)."}</MathBlock></div>

<Prose>{"Write the expression in parentheses as "}<InlineMath>{"z_j"}</InlineMath>{". Hidden state zero contributes weight 1; hidden state one contributes "}<InlineMath>{"e^{z_j}"}</InlineMath>{". Normalizing these two alternatives gives"}</Prose>

<div className="neural-equation"><MathBlock>{"p(h_j=1\\mid v)=\\frac{e^{z_j}}{1+e^{z_j}}=\\sigma(z_j)."}</MathBlock></div>

<Prose>{"The function "}<InlineMath>{"\\sigma(z)=1/(1+e^{-z})"}</InlineMath>{" is the logistic sigmoid. There is no term coupling two hidden switches after "}<InlineMath>{"v"}</InlineMath>{" is fixed, so all their conditional probabilities factorize:"}</Prose>

<div className="neural-equation"><MathBlock>{"p(h\\mid v)=\\prod_j p(h_j\\mid v),\\qquad\np(v_i=1\\mid h)=\\sigma\\!\\left(a_i+\\sum_jW_{ij}h_j\\right)."}</MathBlock></div>

<Prose>{"We can therefore sample an entire hidden layer in parallel given the visible layer, then an entire visible layer given the hidden layer. Sampling a Bernoulli variable with probability 0.7 means drawing a fresh uniform number "}<InlineMath>{"u\\in[0,1)"}</InlineMath>{" and setting the state to one when "}<InlineMath>{"u<0.7"}</InlineMath>{". The number 0.7 is a probability, not a possible state of that binary variable."}</Prose>

<Prose>{""}<strong>{"Conditionally independent does not mean marginally independent."}</strong>{" Once we average over an unknown hidden layer, visible switches can become dependent. Think of two lamps driven by an unobserved common switch. Learning that one is on changes your belief about the common switch, which changes your expectation of the other lamp. This is a probabilistic analogy, not a statement that an undirected RBM identifies causes."}</Prose>

<H3>{"A three-switch model you can completely inspect"}</H3>

<Prose>{"Use two visible switches and one hidden switch. Set both visible biases and the hidden bias to zero, and both weights to "}<InlineMath>{"\\log3"}</InlineMath>{"."}</Prose>

<NeuralTable caption={"A three-switch model you can completely inspect"} headers={[<>{"Visible state"}</>,<>{"Weight with "}<InlineMath>{"h=0"}</InlineMath>{""}</>,<>{"Weight with "}<InlineMath>{"h=1"}</InlineMath>{""}</>,<>{"Sum over hidden state"}</>,<>{"Visible probability"}</>]} rows={[[<>{"00"}</>,<>{"1"}</>,<>{"1"}</>,<>{"2"}</>,<>{"0.1"}</>],[<>{"01"}</>,<>{"1"}</>,<>{"3"}</>,<>{"4"}</>,<>{"0.2"}</>],[<>{"10"}</>,<>{"1"}</>,<>{"3"}</>,<>{"4"}</>,<>{"0.2"}</>],[<>{"11"}</>,<>{"1"}</>,<>{"9"}</>,<>{"10"}</>,<>{"0.5"}</>]]} />

<Prose>{"All eight joint-state weights sum to 20. For visible state 11, "}<InlineMath>{"p(h=1\\mid11)=9/10"}</InlineMath>{". For 10 or 01 it is "}<InlineMath>{"3/4"}</InlineMath>{", and for 00 it is "}<InlineMath>{"1/2"}</InlineMath>{". The hidden state's marginal probability is "}<InlineMath>{"(1+3+3+9)/20=0.8"}</InlineMath>{"."}</Prose>

<Prose>{"Given "}<InlineMath>{"h=0"}</InlineMath>{", each visible switch has on-probability 0.5. Given "}<InlineMath>{"h=1"}</InlineMath>{", it has on-probability 0.75. Consequently each visible marginal is "}<InlineMath>{"0.2(0.5)+0.8(0.75)=0.7"}</InlineMath>{". But "}<InlineMath>{"p(11)=0.5\\ne0.7^2"}</InlineMath>{": the visible variables are dependent despite having no direct edge."}</Prose>

<Prose>{""}<strong>{"Pause:"}</strong>{" if both weights become zero while all biases remain zero, does the hidden switch still create visible dependence?"}</Prose>

<details><summary>Reveal the reasoning</summary>

<Prose>{"No. All eight joint configurations have equal weight. Every visible state has probability "}<InlineMath>{"1/4"}</InlineMath>{", and both visible switches are independent fair Bernoulli variables. A hidden unit with no interaction cannot communicate evidence."}</Prose>

</details>

<H2>{"3. Free energy, exact normalization and what remains difficult"}</H2>

<Prose>{"Observed data contain "}<InlineMath>{"v"}</InlineMath>{", not "}<InlineMath>{"h"}</InlineMath>{". We must add the probability of every hidden explanation, not choose the single best explanation. Define "}<strong>{"free energy"}</strong>{" by"}</Prose>

<div className="neural-equation"><MathBlock>{"e^{-F(v)}=\\sum_h e^{-E(v,h)}."}</MathBlock></div>

<Prose>{"For this RBM the sum factors:"}</Prose>

<div className="neural-equation"><MathBlock>{"e^{-F(v)}=e^{a^Tv}\\prod_j\\left(1+e^{b_j+v^TW_{:,j}}\\right),"}</MathBlock></div>

<Prose>{"so"}</Prose>

<div className="neural-equation"><MathBlock>{"F(v)=-a^Tv-\\sum_j\\operatorname{softplus}\\left(b_j+v^TW_{:,j}\\right),\n\\quad\\operatorname{softplus}(z)=\\log(1+e^z)."}</MathBlock></div>

<Prose>{"Each factor adds the two possible states of one hidden switch. Multiplying those sums accounts for every hidden combination without enumerating them individually. In our example "}<InlineMath>{"e^{-F(11)}=10"}</InlineMath>{" and "}<InlineMath>{"e^{-F(10)}=4"}</InlineMath>{"; the probability ratio is "}<InlineMath>{"10/4=2.5"}</InlineMath>{" within this one model."}</Prose>

<Prose>{"The product is also a way to see why several hidden features can jointly constrain a pattern. This connection is discussed as a product of experts in "}<a href={"https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf"}>{"Hinton's original technical report"}</a>{". It differs from choosing one expert in a mixture; the RBM factors all contribute to the same visible configuration."}</Prose>

<Prose>{"Computing "}<InlineMath>{"F(v)"}</InlineMath>{" is inexpensive, but a normalized log probability still needs"}</Prose>

<div className="neural-equation"><MathBlock>{"\\log p(v)=-F(v)-\\log Z."}</MathBlock></div>

<Prose>{"For "}<InlineMath>{"D"}</InlineMath>{" binary visible variables, there are "}<InlineMath>{"2^D"}</InlineMath>{" visible configurations. This is why exact normalization becomes difficult in many useful RBMs. The restriction simplifies conditionals and marginalizes one layer efficiently; it does not make every global sum cheap."}</Prose>

<H3>{"Enumerate the smaller layer"}</H3>

<Prose>{"There is a valuable exception. Instead of enumerating visible vectors, sum them out and enumerate hidden vectors:"}</Prose>

<div className="neural-equation"><MathBlock>{"Z=\\sum_{h\\in\\{0,1\\}^H}e^{b^Th}\n\\prod_{i=1}^{D}\\left(1+e^{a_i+(Wh)_i}\\right)."}</MathBlock></div>

<Prose>{"Our experiment has 64 visible variables but only eight hidden variables. There are "}<InlineMath>{"2^8=256"}</InlineMath>{" hidden configurations. For each one, a 64-term product represents the sum over all visible configurations. We can compute an exact normalizer and exact model expectations without visiting "}<InlineMath>{"2^{64}"}</InlineMath>{" images."}</Prose>

<Prose>{"“Exact” here describes the finite sum being evaluated; ordinary floating-point rounding remains. Eight hidden units were chosen to make the comparison inspectable, not because they are a universal best size. Raising "}<InlineMath>{"H"}</InlineMath>{" from 8 to 24 multiplies the enumeration count by "}<InlineMath>{"2^{16}=65,536"}</InlineMath>{"."}</Prose>

<Prose>{"Use stable logarithmic calculations. Computing "}<code>{"log(1 + exp(z))"}</code>{" directly can overflow even when its mathematical answer is finite. "}<code>{"numpy.logaddexp(0, z)"}</code>{" evaluates softplus stably, and "}<code>{"scipy.special.logsumexp"}</code>{" handles sums of exponentials in log space. Merely replacing "}<code>{"log"}</code>{" with "}<code>{"log1p"}</code>{" does not fix overflow in an already computed "}<code>{"exp(z)"}</code>{"."}</Prose>

<RbmEnumerationFigure />

<H2>{"4. Learning: which co-occurrences need more probability?"}</H2>

<Prose>{"We want observed examples to receive more probability. For one weight, differentiating the log probability gives"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{\\partial\\log p(v)}{\\partial W_{ij}}\n=v_i\\,p(h_j=1\\mid v)-\\mathbb E_{p(v',h')}[v'_ih'_j]."}</MathBlock></div>

<Prose>{"The first term asks how often that visible/hidden pair is on together when the visible state is supplied by the data. The second asks how often the pair is on together under the model's own distribution. Training increases a weight when the data require more of that co-occurrence than the model currently supplies."}</Prose>

<Prose>{"For a minibatch, average the first term across examples. The corresponding bias gradients are"}</Prose>

<div className="neural-equation"><MathBlock>{"\\nabla_a\\log p=\\mathbb E_{\\rm data}[v]-\\mathbb E_{\\rm model}[v],\n\\qquad\\nabla_b\\log p=\\mathbb E_{\\rm data}[p(h=1\\mid v)]-\\mathbb E_{\\rm model}[h]."}</MathBlock></div>

<Prose>{"The names "}<strong>{"positive phase"}</strong>{" and "}<strong>{"negative phase"}</strong>{" refer to these two expectations. They do not mean that positive examples are labeled correct and negative examples are labeled incorrect. An unsupervised RBM needs no class label in this objective."}</Prose>

<Prose>{"Why subtract the model term? Increasing a weight changes probabilities across the entire state space. The derivative of "}<InlineMath>{"\\log Z"}</InlineMath>{" accounts for that competition. If we only lowered energies of data examples, we could also lower many unwanted states and mistake growing unnormalized scores for improving probability."}</Prose>

<H3>{"One fully checked update"}</H3>

<Prose>{"Suppose the training observation is 11 in our three-switch model. Each positive weight statistic is "}<InlineMath>{"1(0.9)=0.9"}</InlineMath>{". Each model statistic is"}</Prose>

<div className="neural-equation"><MathBlock>{"p(h=1)\\,p(v_i=1\\mid h=1)=0.8(0.75)=0.6."}</MathBlock></div>

<Prose>{"Both weight gradients are 0.3. Both visible-bias gradients are "}<InlineMath>{"1-0.7=0.3"}</InlineMath>{", and the hidden-bias gradient is "}<InlineMath>{"0.9-0.8=0.1"}</InlineMath>{"."}</Prose>

<Prose>{"With learning rate 0.1, update "}<strong>{"all parameters from the same old state"}</strong>{":"}</Prose>

<NeuralTable caption={"One fully checked update"} headers={[<>{"Parameter"}</>,<>{"Before"}</>,<>{"After"}</>]} rows={[[<>{"Each interaction"}</>,<>{""}<InlineMath>{"\\log3"}</InlineMath>{""}</>,<>{""}<InlineMath>{"\\log3+0.03"}</InlineMath>{""}</>],[<>{"Each visible bias"}</>,<>{"0"}</>,<>{"0.03"}</>],[<>{"Hidden bias"}</>,<>{"0"}</>,<>{"0.01"}</>]]} />

<Prose>{"Recomputing the complete distribution changes "}<InlineMath>{"\\log p(11)"}</InlineMath>{" from −0.69314718 to −0.65690173. The observation has become more probable. We checked every analytic gradient against scalar central differences; the largest discrepancy was below "}<InlineMath>{"2\\times10^{-11}"}</InlineMath>{". A bigger learning rate would not automatically preserve improvement."}</Prose>

<RbmGradientLab />

<H3>{"Do we need to sample the positive hidden variables?"}</H3>

<Prose>{"No: for a fixed binary data vector, "}<InlineMath>{"v_i p(h_j=1\\mid v)"}</InlineMath>{" is the exact conditional expectation. Using the probability reduces avoidable sampling noise. This is different from pretending all uncertain variables can be replaced by their means throughout a nonlinear chain. We will test that distinction next."}</Prose>

<H2>{"5. Gibbs sampling, contrastive divergence and persistence"}</H2>

<Prose>{"For larger RBMs, exact model expectations can be impractical. A "}<strong>{"Gibbs sampler"}</strong>{" alternates two easy draws:"}</Prose>

<div className="neural-equation"><MathBlock>{"v^{(0)}\\ \\longrightarrow\\ h^{(0)}\\sim p(h\\mid v^{(0)})\n\\ \\longrightarrow\\ v^{(1)}\\sim p(v\\mid h^{(0)})\\ \\longrightarrow\\cdots."}</MathBlock></div>

<Prose>{"This sequence is a Markov chain: its next state depends on its present state. With finite parameters, the binary RBM's conditional probabilities are strictly between zero and one; its alternating chain can eventually reach every state and has the model distribution as its stationary distribution. How quickly it approaches that distribution is the "}<strong>{"mixing"}</strong>{" question. A sampler may spend many steps near one region even though it is mathematically able to reach another."}</Prose>

<Prose>{"Neither these conditional draws nor a within-model probability ratio requires "}<InlineMath>{"Z"}</InlineMath>{". Difficult normalization and slow mixing are related computational concerns, not the same operation. Introducing a temperature changes the distribution; ordinary sampling at our fixed temperature one does not require an annealing schedule."}</Prose>

<H3>{"CD-\\(k\\): start near a data example"}</H3>

<Prose>{""}<strong>{"Contrastive divergence"}</strong>{", CD-"}<InlineMath>{"k"}</InlineMath>{", starts the chain at data, takes "}<InlineMath>{"k"}</InlineMath>{" full hidden/visible transitions, and uses the resulting visible states for an approximate negative statistic. After the final visible draw, calculate the hidden probabilities again for that statistic. Treat the sampled state as fixed when forming this update; this is not backpropagation through discrete draws."}</Prose>

<Prose>{"We can see the approximation without Monte Carlo noise in the three-switch model. Start at 11. The hidden draw is one with probability 0.9. If it is zero, the four visible outcomes each have probability 0.25. If it is one, their probabilities are "}<InlineMath>{"[1,3,3,9]/16"}</InlineMath>{". Combining the cases gives"}</Prose>

<div className="neural-equation"><MathBlock>{"q_1=[0.08125,\\ 0.19375,\\ 0.19375,\\ 0.53125]."}</MathBlock></div>

<Prose>{"Compare that with the stationary model distribution "}<InlineMath>{"[0.1,0.2,0.2,0.5]"}</InlineMath>{". The chain retains too much probability on its initial state after one transition. Its expected negative weight statistic is 0.6234375, so expected CD-1 gives a weight update direction of "}<InlineMath>{"0.9-0.6234375=0.2765625"}</InlineMath>{", instead of the exact 0.3."}</Prose>

<NeuralTable caption={"CD-\\(k\\): start near a data example"} headers={[<>{"Full transitions"}</>,<>{"Probability of 11"}</>,<>{"Expected weight gradient"}</>,<>{"Distance from stationary distribution*"}</>]} rows={[[<>{"0"}</>,<>{"1"}</>,<>{"0"}</>,<>{"0.5"}</>],[<>{"1"}</>,<>{"0.53125"}</>,<>{"0.2765625"}</>,<>{"0.03125"}</>],[<>{"2"}</>,<>{"0.5029296875"}</>,<>{"0.2978027344"}</>,<>{"0.0029296875"}</>],[<>{"3"}</>,<>{"0.5002746582"}</>,<>{"0.2997940063"}</>,<>{"0.0002746582"}</>]]} />

<Prose>{"*Total variation distance is half the sum of absolute differences between matching state probabilities. These entries come from multiplying an exactly enumerated four-state transition matrix, with parameters held fixed. They are not measured frequencies of four individual samples, or a general promise that three steps suffice."}</Prose>

<RbmChainLab />

<Prose>{"Finite-step CD is a biased approximation to the likelihood gradient. Its usual update also omits a term arising from the parameter dependence of the reconstructed distribution in the proposed CD objective. It need not be the gradient of any scalar objective in general; "}<a href={"https://proceedings.mlr.press/v9/sutskever10a.html"}>{"Sutskever and Tieleman, 2010"}</a>{" analyze that distinction. The successful tiny example above does not establish global convergence."}</Prose>

<H3>{"PCD: keep the model's particles alive"}</H3>

<Prose>{""}<strong>{"Persistent contrastive divergence"}</strong>{", also called stochastic maximum likelihood in this setting, retains a collection of sampled states across parameter updates. Each minibatch supplies fresh positive statistics, while these persistent particles take more Gibbs steps for the negative statistics. They are not reset to the current data minibatch. The hope is that a chain already near the old model remains useful as parameters move. The method and its empirical motivation are described in "}<a href={"https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/Tieleman.2008.pdf"}>{"Tieleman's 2008 paper"}</a>{"."}</Prose>

<Prose>{"Persistence does not magically produce independent equilibrium samples. A rapidly changing model, strongly separated regions or too few particles can still produce poor estimates. CD-"}<InlineMath>{"k"}</InlineMath>{" trades additional transitions for cost; PCD changes initialization and reuse. Neither is universally better for every budget, initialization and evaluation measure."}</Prose>

<RbmPersistenceLab />

<H3>{"Why a mean is not a Gibbs state"}</H3>

<Prose>{"After our first transition, each visible mean is 0.725. Feeding "}<InlineMath>{"[0.725,0.725]"}</InlineMath>{" through the hidden sigmoid gives approximately 0.831036. But averaging the hidden sigmoid across the actual four-state distribution gives 0.809375. The sigmoid of a mean is not generally the mean of a sigmoid."}</Prose>

<Prose>{"Using visible probabilities in reconstruction can be a useful explicitly labeled heuristic. It is not an exact transition of this binary Gibbs chain. Hinton's "}<a href={"https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf"}>{"practical guide"}</a>{" discusses probability/state choices and practical monitoring; read those choices in the context of the intended objective. Our training sampler draws binary visible states, while our reconstruction diagnostic deliberately uses a deterministic mean-to-mean pass."}</Prose>

<H2>{"6. A real experiment whose likelihood we can actually calculate"}</H2>

<Prose>{"We use real handwritten digit images from "}<a href={"https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits"}>{"UCI Optical Recognition of Handwritten Digits"}</a>{", by E. Alpaydin and C. Kaynak, distributed under CC BY 4.0. This is the small 8×8 dataset available through scikit-learn, not MNIST. The retained CSV contains the first 40 occurrences of each digit in that loader, with original one-based row identifiers and 64 pixel values from 0 to 16."}</Prose>

<Prose>{"The task is "}<strong>{"unsupervised probability modeling of binary images"}</strong>{". Turn a pixel on when its value is at least eight. This deterministic transformation discards grayscale information and defines what the model's likelihood means. A likelihood for these binary images is not a likelihood for the original grayscale measurements."}</Prose>

<Prose>{"After binarization, source 228 duplicates source 12, and source 300 duplicates source 274. Keep the first occurrence before splitting. The remaining 398 distinct binary images are divided within digit using seed 91: eight per class for development, eight for assessment, and the remaining 238 for fitting. Labels are used only for this stratification; they never enter the RBM loss. The source IDs and transformations are in "}<a href={"/learn-code/boltzmann-machines-restricted-boltzmann-machines-rbm/data-provenance.md"}>{"data-provenance.md"}</a>{"."}</Prose>

<Prose>{"Writer identifiers are unavailable, so this is not a writer-independent evaluation. These images have also appeared in other lessons. Treat this as a reproducible classroom experiment, not a newly untouched benchmark or a comparison with published UCI scores."}</Prose>

<H3>{"Protocol before results"}</H3>

<Prose>{"Compare a 64-parameter independent-pixel Bernoulli baseline with three training procedures for the same 64-visible, eight-hidden RBM, which has 584 parameters. The baseline's pixel probabilities are "}<InlineMath>{"(n_{\\rm on}+0.5)/(n_{\\rm fit}+1)"}</InlineMath>{". This small symmetric smoothing avoids infinite logits at always-off pixels. Initialize RBM visible biases to those logits, hidden biases to zero, and interactions to small independent normal values with standard deviation 0.01."}</Prose>

<Prose>{"Run exact likelihood-gradient ascent, sampled CD-1, and sampled PCD-1 for seeds 11, 29 and 47. Each uses 300 epochs, batch size 64 and fixed step size 0.05, without momentum, weight decay, dropout or checkpoint selection. Corresponding seeds share initial parameters and minibatch orders. PCD keeps 64 particles; the final positive minibatch has 46 examples, so positive and negative means are normalized separately. No tuning used development or assessment outcomes."}</Prose>

<Prose>{"We evaluate exact "}<strong>{"negative log likelihood"}</strong>{", NLL, in natural-log units, or "}<strong>{"nats per image"}</strong>{". Lower is better. A model assigning probability "}<InlineMath>{"p"}</InlineMath>{" to an image incurs "}<InlineMath>{"-\\log p"}</InlineMath>{". The average concerns the actual binary images, not whether a nearest-looking reconstruction was produced."}</Prose>

<NeuralTable caption={"Protocol before results"} headers={[<>{"Method"}</>,<>{"Seed 11 assessment NLL"}</>,<>{"Seed 29"}</>,<>{"Seed 47"}</>]} rows={[[<>{"Independent pixels"}</>,<>{"24.311674"}</>,<>{"Same fixed model"}</>,<>{"Same fixed model"}</>],[<>{"Exact gradient"}</>,<>{"20.496861"}</>,<>{"20.655667"}</>,<>{"20.531889"}</>],[<>{"CD-1"}</>,<>{"20.556534"}</>,<>{"20.682717"}</>,<>{"20.575317"}</>],[<>{"PCD-1"}</>,<>{"20.534554"}</>,<>{"20.751721"}</>,<>{"20.568239"}</>]]} />

<Prose>{"All nine fitted RBMs improved on the independent baseline in this run. Exact-gradient training obtained the lowest assessment NLL for each matched seed, but CD-1 and PCD-1 trade places. Do not turn three seeds on 80 assessment images into a universal ranking. Nor does an exact gradient guarantee a global optimum or better generalization for every initialization."}</Prose>

<Prose>{"For exact-gradient seed 11, fit/development/assessment NLLs are 19.791466, 20.650057 and 20.496861. The gap is a reason to monitor generalization, not evidence that every higher-capacity model necessarily fails. Full histories, per-image NLLs and all final weights are retained in "}<a href={"/learn-code/boltzmann-machines-restricted-boltzmann-machines-rbm/calculated-inputs.json"}>{"calculated-inputs.json"}</a>{"."}</Prose>

<RbmStudyFigure />

<H3>{"Complete runnable study"}</H3>

<Prose>{"Save the following as "}<a href={"/learn-code/boltzmann-machines-restricted-boltzmann-machines-rbm/rbm-study.py"}>{"rbm-study.py"}</a>{", next to "}<a href={"/learn-code/boltzmann-machines-restricted-boltzmann-machines-rbm/digits-400.csv"}>{"digits-400.csv"}</a>{", and run "}<code>{"python rbm-study.py"}</code>{". It needs Python, NumPy and SciPy; the recorded execution used Python 3.12.14 and NumPy 2.3.5 on CPU. The program defines the model, preprocessing, roles, all nine fits, exact checks and completion calculations. It does not download a dataset or a pretrained model. Matrix products use batch rows, "}<InlineMath>{"W"}</InlineMath>{" stores visible-by-hidden weights, and updates are ascent because we maximize log probability."}</Prose>

<RbmProgram file="rbm-study.py" />

<Prose>{"The program writes its computed evidence beside itself. To reproduce the report, retain that output and the data provenance. For a new experiment, choose the split, hyperparameter search and final assessment procedure before comparing results; changing code after seeing this assessment set does not restore its independence."}</Prose>

<H2>{"7. Reconstructing, sampling and completing are different tasks"}</H2>

<H3>{"A convincing reconstruction can hide a poor distribution"}</H3>

<Prose>{"Our deterministic reconstruction computes "}<InlineMath>{"\\hat v=\\sigma(a+W\\sigma(b+W^Tv))"}</InlineMath>{", with vector orientation adjusted in code. It asks how closely a two-stage mean pass returns the input. This is useful to inspect, but its mean squared error is not NLL."}</Prose>

<Prose>{"A constructed two-pixel counterexample makes the distinction sharp. Model A has both interactions 20, visible biases −10 and hidden bias −20. It puts almost half its probability on 00 and half on 11. Starting from 11, the deterministic reconstruction is almost exactly 11: per-pixel MSE is about "}<InlineMath>{"2.06\\times10^{-9}"}</InlineMath>{". Nevertheless NLL for 11 is about 0.693238."}</Prose>

<Prose>{"Model B has no interactions and both visible on-probabilities 0.9. Its reconstruction of 11 is "}<InlineMath>{"[0.9,0.9]"}</InlineMath>{", with the larger MSE 0.01. But it assigns probability 0.81 to 11, giving the "}<strong>{"better"}</strong>{" NLL 0.210721. Reconstruction measures the return journey near an input; likelihood measures how the model divides its full probability budget."}</Prose>

<RbmReconstructionLab />

<Prose>{"Similarly, comparing free energies from different model versions requires their different normalizers. A constant downward energy shift could make every raw score look better without changing any probability. Under the same fixed model, a data-versus-development mean free-energy gap does equal the corresponding NLL gap because the one "}<InlineMath>{"\\log Z"}</InlineMath>{" cancels. That useful diagnostic does not license comparing raw free energy across epochs as if it were normalized likelihood."}</Prose>

<H3>{"Generate without showing the model an input"}</H3>

<Prose>{"Ordinary Gibbs generation runs a chain and must consider burn-in and mixing. Our eight-hidden-unit model has an additional option: enumerate the exact hidden marginal, draw one of its 256 states, and independently draw visible pixels conditional on it. Repeating that process produces independent model samples, apart from pseudorandom-number implementation details, without a burn-in chain."}</Prose>

<Prose>{"The saved 16 samples for every fitted model use this exact-enumeration route. They are not reconstructions of selected assessment images. Show all 16 in fixed order, including odd-looking ones. Eight hidden bits support at most 256 hidden configurations, but that does "}<strong>{"not"}</strong>{" limit the generator to 256 visible images: each hidden configuration produces a full Bernoulli distribution over pixels."}</Prose>

<RbmSampleGallery />

<H3>{"Complete missing pixels without treating missing as black"}</H3>

<Prose>{"Suppose the left half of an image is observed and the right half is unknown. “Unknown” is not a measured zero. Let "}<InlineMath>{"O"}</InlineMath>{" contain observed pixel indices and "}<InlineMath>{"M"}</InlineMath>{" the missing ones. For each hidden state, its log weight given the observed values is"}</Prose>

<div className="neural-equation"><MathBlock>{"b^Th+\\sum_{i\\in O}v_i\\big(a_i+(Wh)_i\\big)\n+\\sum_{i\\in M}\\operatorname{softplus}\\big(a_i+(Wh)_i\\big)."}</MathBlock></div>

<Prose>{"Normalize those 256 weights to obtain "}<InlineMath>{"p(h\\mid v_O)"}</InlineMath>{". A missing pixel's conditional on-probability is then"}</Prose>

<div className="neural-equation"><MathBlock>{"p(v_i=1\\mid v_O)=\\sum_h p(h\\mid v_O)\\sigma\\big(a_i+(Wh)_i\\big)."}</MathBlock></div>

<Prose>{"The first sum fixes observed states; the softplus sum marginalizes unknown ones. It is the same reasoning that produced "}<InlineMath>{"Z"}</InlineMath>{", now with some variables held fixed. This exact calculation is practical because the hidden layer is small. With a larger hidden layer, a clamped Gibbs chain is an approximate alternative: after each visible draw restore the observed pixels, while allowing missing pixels to change. Do not continually clamp the missing pixels to their initialization."}</Prose>

<Prose>{"For the three-switch example, observing "}<InlineMath>{"v_1=1"}</InlineMath>{" gives "}<InlineMath>{"p(v_2=1\\mid v_1=1)=0.5/(0.2+0.5)=5/7"}</InlineMath>{". Filling the missing value with zero and treating it as observed would give a different hidden posterior. A placeholder is a display decision, not evidence."}</Prose>

<Prose>{"We applied the fixed left-half mask to every assessment image. There are "}<InlineMath>{"80\\times32=2,560"}</InlineMath>{" missing pixel targets. Exact-gradient seed 11 achieved conditional-probability MSE 0.117704 versus the independent baseline's 0.132506; thresholding at 0.5 gives 2,112 versus 2,028 correct missing pixels. These are pixel-level results, not digit-classification accuracy. Other seeds and methods are retained, including PCD-1 seed 47's MSE 0.116945. A model can do well at this conditional task without winning on full-image NLL."}</Prose>

<RbmDigitLab />

<Prose>{"The conditional mean can look blurry because several plausible completions disagree. It is not necessarily a plausible joint sample. To see whole alternatives, sample a hidden state from the observed-data posterior and then all missing pixels conditional on it, preserving the observed pixels exactly. Uncertainty is part of the answer."}</Prose>

<H2>{"8. Broader uses and deeper boundaries"}</H2>

<H3>{"Features, labels and ratings"}</H3>

<Prose>{"The vector "}<InlineMath>{"p(h=1\\mid v)"}</InlineMath>{" provides learned features that can feed a classifier. To evaluate this use, fit the RBM and any preprocessing only on permitted training inputs, then fit the classifier using training labels and assess on held-out data. That is a separate supervised experiment; our unsupervised NLL table supplies no classification result."}</Prose>

<Prose>{"A different construction includes a categorical label as an additional visible variable. Evaluate the joint free energy for each possible label and normalize over those alternatives to obtain "}<InlineMath>{"p(y\\mid v)"}</InlineMath>{". The one global "}<InlineMath>{"Z"}</InlineMath>{" cancels. In contrast, separate class-specific RBMs have separate "}<InlineMath>{"Z_c"}</InlineMath>{" values: simply choosing the lowest raw free energy can be wrong. You must account for normalization and class priors or use a properly fitted discriminative calibration."}</Prose>

<Prose>{"Movie ratings provide a historically important application beyond images. The "}<a href={"https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf"}>{"2007 collaborative-filtering paper"}</a>{" used categorical rating units and shared parameters across user-specific models containing their observed movies. Its treatment of absent ratings is a family of models with tied parameters, not equivalent to inserting zeros or exactly marginalizing every unrated movie in a single fixed model. The useful lesson is to specify both the rating distribution and what “missing” means before choosing the learning rule. This is historical methodology, not a claim that an RBM is today's best recommender."}</Prose>

<H3>{"Binary is a modeling choice"}</H3>

<Prose>{"Bernoulli variables describe binary events. One-hot categorical variables need probabilities that sum to one within each category group. Continuous measurements may use Gaussian visible variables; count data need a suitable count distribution. Changing the support requires changing the energy and conditional distributions consistently. Dividing a real value into "}<InlineMath>{"[0,1]"}</InlineMath>{" does not by itself turn its likelihood into a Bernoulli probability for that real value."}</Prose>

<Prose>{"For a simple fixed-unit-variance Gaussian-visible, binary-hidden model,"}</Prose>

<div className="neural-equation"><MathBlock>{"E(v,h)=\\tfrac12\\|v-a\\|^2-b^Th-v^TWh,"}</MathBlock></div>

<Prose>{"completing the square gives "}<InlineMath>{"v\\mid h\\sim\\mathcal N(a+Wh,I)"}</InlineMath>{", while "}<InlineMath>{"p(h_j=1\\mid v)=\\sigma(b_j+v^TW_{:,j})"}</InlineMath>{". The quadratic term prevents arbitrarily large visible values from obtaining unbounded preference for a fixed hidden state. If both layers are continuous, interaction strength and the full quadratic form must support a finite normalizer. An ordinary neural activation substituted into a sampler is not enough to define a valid probability model."}</Prose>

<H3>{"Stacking does not erase the model definition"}</H3>

<Prose>{"An RBM has one bipartite undirected layer pair. A "}<strong>{"deep belief network"}</strong>{" uses an undirected top pair with directed conditional layers below it; the original "}<a href={"https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf"}>{"2006 DBN paper"}</a>{" develops layerwise initialization and subsequent learning. A "}<strong>{"deep Boltzmann machine"}</strong>{" keeps its multilayer interactions undirected and has harder hidden inference, as described in "}<a href={"https://proceedings.mlr.press/v5/salakhutdinov09a.html"}>{"Salakhutdinov and Hinton, 2009"}</a>{"."}</Prose>

<Prose>{"RBMs can initialize other networks, but a stack of independently trained RBMs is not automatically the product of their joint distributions with correct normalization. A deterministic encoder fine-tuned for reconstruction becomes an autoencoder with its own objective. A classifier fine-tuned for labels is evaluated on conditional prediction. Draw the final generative graph and write its probability factorization before transferring a statement about one model to another."}</Prose>

<RbmFamilyFigure />

<Prose>{"These models helped develop representation-learning ideas. Their history is useful without declaring that unsupervised pretraining always helps, or that energy-based modeling has disappeared. Modern architectures, data regimes and objectives require their own evidence."}</Prose>

<H3>{"Beyond exact enumeration: AIS and other objectives"}</H3>

<Prose>{"For a large RBM, "}<strong>{"annealed importance sampling"}</strong>{", AIS, can estimate a partition-function ratio by traversing intermediate distributions between a tractable base and the target. Each stage contributes a ratio of unnormalized weights, and its transition must preserve the intended intermediate distribution. The "}<a href={"https://www.cs.toronto.edu/~rsalakhu/papers/dbn_ais.pdf"}>{"2008 quantitative analysis paper"}</a>{" applies this approach to RBMs and DBNs."}</Prose>

<Prose>{"Under the required support, initialization and transition assumptions, the average importance weight estimates the normalizer ratio without bias. Taking its logarithm is a nonlinear operation; a log estimate is generally biased and a particular run is not a certified lower or upper bound. Report repeats, weight variability, schedules and assumptions. Our small example avoids this estimation issue by summing exactly; it should not teach an AIS estimate as an exact NLL curve."}</Prose>

<RbmAisFigure />

<Prose>{"Other training criteria answer different questions. Pseudo-likelihood uses conditionals such as "}<InlineMath>{"p(v_i\\mid v_{-i})"}</InlineMath>{" and can avoid the global normalizer. Current "}<a href={"https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html"}>{"scikit-learn BernoulliRBM documentation"}</a>{" identifies its training as PCD/SML and its "}<code>{"score_samples"}</code>{" result as a random-bit pseudo-likelihood estimate, not exact log likelihood. Check that metric contract before plotting a library “score” beside our NLL."}</Prose>

<Prose>{"Likewise an autoencoder reconstruction loss, a variational lower bound and a normalized likelihood are distinct quantities. A comparison becomes meaningful only after specifying the data representation, objective, evaluation measure and computational budget."}</Prose>

<H3>{"Use a fitted library RBM without changing the probability question"}</H3>

<Prose>{"The scratch study owns the probability model: "}<code>{"free_energy"}</code>{", "}<code>{"positive"}</code>{", "}<code>{"exact_negative"}</code>{", "}<code>{"gibbs"}</code>{" and "}<code>{"conditional_missing"}</code>{" in "}<a href={"/learn-code/boltzmann-machines-restricted-boltzmann-machines-rbm/rbm-study.py"}>{"rbm-study.py"}</a>{". Its model-weight update in "}<code>{"main"}</code>{" is simultaneous ascent from old-state statistics. Exact enumeration is the deliberately small oracle, while CD and PCD supply sampled negative statistics. We now keep those equations and inspect the model learned by the ordinary library."}</Prose>

<Prose>{"The "}<a href={"/learn-code/boltzmann-machines-restricted-boltzmann-machines-rbm/bernoulli_rbm_bridge.py"}>{"complete bridge"}</a>{" uses scikit-learn 1.9.1. Keep it beside "}<code>{"rbm-study.py"}</code>{" and run "}<code>{"python bernoulli_rbm_bridge.py"}</code>{" in the earlier environment with "}<code>{"scikit-learn==1.9.1"}</code>{" installed. These two filenames have an actual dependency: the bridge imports the scratch free-energy and normalization functions without launching the digit experiment. The four constructed fitting patterns have frequencies 1, 2, 2 and 5; this is a compact API demonstration, not another assessment experiment."}</Prose>

<NeuralTable caption={"Use a fitted library RBM without changing the probability question"} headers={[<>{"Equation or state"}</>,<>{"Ordinary API counterpart"}</>,<>{"Meaning that must survive"}</>]} rows={[[<>{"W with visible rows and hidden columns"}</>,<>{""}<code>{"components_.T"}</code>{""}</>,<>{"The transpose is required; the library stores hidden-by-visible weights"}</>],[<>{"a, b"}</>,<>{""}<code>{"intercept_visible_"}</code>{", "}<code>{"intercept_hidden_"}</code>{""}</>,<>{"Visible and hidden biases are not interchangeable"}</>],[<>{"σ(vW+b)"}</>,<>{""}<code>{"transform(v)"}</code>{""}</>,<>{"Hidden probabilities, not sampled binary states"}</>],[<>{"Alternating hidden/visible draws"}</>,<>{""}<code>{"gibbs(v)"}</code>{""}</>,<>{"One visible-state transition; it does not fit parameters"}</>],[<>{"Persistent negative statistics"}</>,<>{""}<code>{"fit"}</code>{"'s stochastic maximum likelihood updates"}</>,<>{"Package chain initialization, minibatches and random draws need not match our scratch trajectory"}</>],[<>{"−F(v)−log Z"}</>,<>{"Exact tiny oracle in this bridge"}</>,<>{""}<code>{"score_samples"}</code>{" instead estimates random-bit pseudo-likelihood"}</>]]} />

<Prose>{"The first comparison uses "}<strong>{"identical fitted parameters and inputs"}</strong>{", not independently trained models. Then the four-state transition matrix verifies that the fitted distribution is stationary under its alternating Gibbs transition. A single call to "}<code>{"gibbs"}</code>{" is random and need not resemble that distribution on four samples. The printout deliberately keeps hidden probabilities, sampled states, pseudo-likelihood and exact log probability separately named. The API contract and orientation are documented by "}<a href={"https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html"}>{"scikit-learn"}</a>{"."}</Prose>

<RbmProgram file="bernoulli_rbm_bridge.py" />

<Prose>{"A small authoring probe executed these assertions with scikit-learn 1.9.1: the fitted probabilities of 00,01,10,11 were approximately [0.183992, 0.246088, 0.243343, 0.326576], and the same-parameter hidden-probability and stationary-transition checks passed. These are constructed-case author checks, not independent implementation certification or a new digit benchmark. Final source/display verification and publication checks belong to implementation. The previously reported digit results remain unchanged. At batch size B, D visible variables and H hidden variables, one conditional sweep uses O(BDH) arithmetic and O(B(D+H)+DH) working/parameter storage. The tiny probability oracle costs exponentially in H; do not expand it to hundreds of hidden units just to accompany a library fit."}</Prose>

<RbmLibraryLab />

<Prose>{""}<strong>{"Take control."}</strong>{" Increase only the fitted hidden bias by log 2, recompute "}<code>{"transform"}</code>{", exact visible probabilities and the transition matrix, and inspect the changed result. Then change only the fitting seed and distinguish two claims: the matched arithmetic must still agree, but the learned parameters may differ."}</Prose>

<details><summary>Hint and reasoned solution</summary>

<Prose>{"Changing b adds log 2 to every hidden logit, doubling each hidden on/off odds. It does not double a probability, which is bounded by one. Copy the edited "}<code>{"intercept_hidden_"}</code>{" into the scratch b before all comparisons; recompute Z after the edit. The stationary-vector check remains valid for the edited model. Comparing its newly computed free energy with the old Z is an intentionally broken normalization. A new fitting seed changes the experiment, so retain a separate model and perform its own same-parameter comparison rather than requiring two random fits to agree."}</Prose>

</details>

<H2>{"9. Diagnose failures by the quantity that failed"}</H2>

<NeuralTable caption={"9. Diagnose failures by the quantity that failed"} headers={[<>{"Observation"}</>,<>{"What it may mean"}</>,<>{"Next useful check"}</>]} rows={[[<>{"Reconstruction improves; likelihood does not"}</>,<>{"Local return paths improved without the desired probability allocation"}</>,<>{"Calculate exact NLL if feasible; otherwise use a justified normalizer estimate and inspect samples"}</>],[<>{"A persistent chain stays near one image"}</>,<>{"Mixing may be poor; correlated particles can give misleading negative statistics"}</>,<>{"Compare several initial states, autocorrelation and region occupancy; inspect update size"}</>],[<>{"Raw free energies fall each epoch"}</>,<>{"Parameters or offsets changed, not necessarily normalized probability"}</>,<>{"Include the same version's "}<InlineMath>{"\\log Z"}</InlineMath>{""}</>],[<>{"Hidden probabilities are nearly all zero or one"}</>,<>{"Bias/weight scale, saturation or data mismatch may be limiting useful features"}</>,<>{"Inspect shared-scale weights, per-unit activation distributions and actual gradient increments"}</>],[<>{"A grayscale input is silently accepted by a binary sampler"}</>,<>{"API numeric acceptance may not match the stated likelihood"}</>,<>{"Declare deterministic or stochastic binarization, or choose another visible distribution"}</>],[<>{"Filling a missing placeholder changes the answer"}</>,<>{"Missing values were accidentally treated as evidence"}</>,<>{"Verify mask handling against exact tiny conditional calculations"}</>],[<>{"Training works only after using assessment outcomes to tune it"}</>,<>{"Reported assessment is no longer independent"}</>,<>{"Use a fresh final evaluation protocol; keep all attempted outcomes disclosed"}</>]]} />

<Prose>{"Large weights can make sampling difficult without necessarily producing NaNs. Numerically stable softplus avoids one arithmetic failure; it does not cure poor mixing or an inappropriate learning rate. Momentum, decay and sparsity penalties can be investigated, but each changes the update and requires its own declared choice and validation. Add one justified change at a time so you can tell what caused the result."}</Prose>

<section className="lesson-ending lesson-ending--practice" data-lesson-ending="practice"><H2>{"10. Practice: make the probability bookkeeping explicit"}</H2>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"1. Change one preference"}</H3>

<Prose>{"In the three-switch model, increase only "}<InlineMath>{"a_1"}</InlineMath>{" from zero to "}<InlineMath>{"\\log2"}</InlineMath>{". What are the four visible probabilities? Does every state become twice as likely?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Only configurations with "}<InlineMath>{"v_1=1"}</InlineMath>{" acquire the extra factor two. Recompute the normalizer after applying it."}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{"The unnormalized visible weights in order "}<InlineMath>{"00,01,10,11"}</InlineMath>{" become "}<InlineMath>{"2,4,8,20"}</InlineMath>{". Their sum is 34, so probabilities are "}<InlineMath>{"1/17,2/17,4/17,10/17"}</InlineMath>{". Ratios within the "}<InlineMath>{"v_1=1"}</InlineMath>{" group stay the same, but all normalized probabilities must reflect the new total. A global factor would cancel; this was a state-dependent factor."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"2. Change the training observation"}</H3>

<Prose>{"Use the original three-switch parameters but observe 10 instead of 11. Calculate both weight gradients and all three bias gradients. State which interactions increase."}</Prose>

<details><summary>Hint</summary>

<Prose>{"The hidden on-probability is "}<InlineMath>{"3/4"}</InlineMath>{". The model expectations are unchanged until you update the model."}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{"Positive weight statistics are "}<InlineMath>{"[0.75,0]"}</InlineMath>{" and negative statistics remain "}<InlineMath>{"[0.6,0.6]"}</InlineMath>{", giving gradients "}<InlineMath>{"[0.15,-0.6]"}</InlineMath>{". Visible-bias gradients are "}<InlineMath>{"[1,0]-[0.7,0.7]=[0.3,-0.7]"}</InlineMath>{". The hidden-bias gradient is "}<InlineMath>{"0.75-0.8=-0.05"}</InlineMath>{". Only the first interaction increases under positive-step gradient ascent. “Positive phase” does not mean that every parameter moves upward."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"3. Exact sampling without \\(Z\\) in the transition"}</H3>

<Prose>{"Starting from visible 10, use uniform draw 0.8 for the hidden switch, then draws 0.3 and 0.6 for the two visible switches. What is the next visible state? Is that one sample evidence that the stationary probability of that state is one?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Compare each draw with its current conditional probability. A sampled hidden zero changes the visible conditionals."}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{"The hidden on-probability is 0.75, so draw 0.8 gives hidden zero. Both visible on-probabilities are then 0.5. Draws 0.3 and 0.6 produce 10. One transition supplies one random outcome; it neither estimates an entire distribution accurately nor establishes mixing. Repeating a state can occur in a perfectly valid chain."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"4. A different missing observation"}</H3>

<Prose>{"Now observe "}<InlineMath>{"v_1=0"}</InlineMath>{" in the three-switch model and leave "}<InlineMath>{"v_2"}</InlineMath>{" missing. Find "}<InlineMath>{"p(v_2=1\\mid v_1=0)"}</InlineMath>{". Compare it with the case "}<InlineMath>{"v_1=1"}</InlineMath>{" and with the unconditional marginal."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Restrict the visible probability table to the states compatible with the observation, then normalize that smaller table."}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{"The compatible states 00 and 01 have probabilities 0.1 and 0.2, so the answer is "}<InlineMath>{"0.2/0.3=2/3"}</InlineMath>{". Observing one gives "}<InlineMath>{"5/7"}</InlineMath>{"; the unconditional answer is 0.7. The hidden variable induces dependence, so evidence changes the conditional answer. An unknown value is neither of the two observed cases."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"5. Choose the correct metric"}</H3>

<Prose>{"A report claims model A is the better density model because its reconstruction MSE is smaller. For one assessment image, model A assigns probability 0.2 and model B assigns probability 0.3. Which has better NLL on this image? What additional evidence would you request for a useful generator?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Take the negative logarithm. Then separate scoring known observations from sampling plausible and diverse new ones."}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{"The per-image NLLs are "}<InlineMath>{"-\\log0.2\\approx1.6094"}</InlineMath>{" and "}<InlineMath>{"-\\log0.3\\approx1.2040"}</InlineMath>{", so B is better on that measure. This comparison addresses that one image; assess the full held-out set before making a general performance claim. Request the normalization method, data split and representation, complete sampling protocol, unselected samples and relevant diversity/coverage evaluation. MSE alone settles none of those questions."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"6. A resource decision"}</H3>

<Prose>{"You have 20 visible binary units and 40 hidden ones. Which layer should you enumerate for an exact normalizer? How does the answer change for 100 visible units and 12 hidden ones?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Either layer can be summed out analytically when the other is fixed. Enumerate the smaller binary state space."}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{"Enumerate the "}<InlineMath>{"2^{20}=1,048,576"}</InlineMath>{" visible configurations in the first case, computing their free energies, rather than "}<InlineMath>{"2^{40}"}</InlineMath>{" hidden configurations. Enumerate "}<InlineMath>{"2^{12}=4,096"}</InlineMath>{" hidden configurations in the second case. Memory can be bounded by summing in chunks with a stable log accumulator, but the total enumeration work remains exponential in the enumerated layer size."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"7. An implementation diagnosis"}</H3>

<Prose>{"A PCD trainer resets its particles to the current data minibatch at every iteration, replaces sampled visible states by their means, and reports "}<code>{"score_samples"}</code>{" as exact likelihood. Identify three separate issues and specify a repair for each."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Ask where the negative chain starts, what state space its transitions occupy, and what quantity the scoring API returns."}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{"Resetting to data removes persistence; retain chain state across updates or rename and implement an intended CD method. Mean-visible replacements change the Bernoulli Gibbs transition; draw binary states for the stated sampler, or explicitly name and evaluate the heuristic. Scikit-learn's RBM score is a pseudo-likelihood estimate; report that name and use an exact normalizer or documented estimator for normalized likelihood. These are independent mistakes, so repairing only one does not fix the experiment."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"8. Extend the real study without leaking information"}</H3>

<Prose>{"Design a comparison of 8 versus 12 hidden units for missing-pixel completion. State what stays fixed, what is selected using development data, and what must remain unavailable during fitting and selection."}</Prose>

<details><summary>Hint</summary>

<Prose>{"The hidden-state sums are still manageable, but the larger model changes capacity and cost. The withheld pixel values are targets for evaluation, not completion inputs."}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{"Keep the binary preprocessing, duplicate handling, source-group split, mask definition and evaluation metrics fixed. Predeclare learning-rate/epoch candidates and seeds for each size, recording their different parameter and enumeration costs. Fit on the fitting role; select configurations from development NLL or a declared development completion metric. Freeze the choice before evaluating a fresh assessment role. Keep assessment labels and missing pixel values out of model fitting, conditioning and selection. Since this lesson already exposes its assessment outcomes, a stronger new empirical claim needs a new untouched evaluation protocol. Report all declared seeds and include the independent-pixel baseline."}</Prose>

</details></div></section>

<section className="lesson-ending lesson-ending--resources" data-lesson-ending="resources"><H2>{"11. References and another way to learn"}</H2>

<Prose>{"Use "}<a href={"https://www.cs.toronto.edu/~hinton/coursera_lectures.html"}>{"Hinton's creator-hosted 2012 lecture collection"}</a>{" for a spoken route: 11e introduces probability modeling, 12a–12d develop learning and RBMs, 12e covers ratings, and 14a–14e connect features, fine-tuning and real-valued data. The page links the individual recordings; the historical methods should be read with their stated context. Lecture titles and links were verified, without claiming a full viewing of every recording."}</Prose>

<Prose>{"For a practical written route, read the "}<a href={"https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf"}>{"RBM guide"}</a>{" after doing the three-switch arithmetic. Its sections on statistics, monitoring, initialization and visible-unit choices help distinguish training decisions. For the reason finite-step CD needs care, follow the "}<a href={"https://proceedings.mlr.press/v9/sutskever10a.html"}>{"convergence paper"}</a>{"; for persistence, follow "}<a href={"https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/Tieleman.2008.pdf"}>{"Tieleman's original method"}</a>{". The "}<a href={"https://www.cs.toronto.edu/~rsalakhu/papers/dbn_ais.pdf"}>{"AIS analysis"}</a>{" is the next step when exact enumeration becomes too large."}</Prose>

<Prose>{"The next topic in this module is "}<a href={"/learn/path/full-curriculum/spectral-normalization-gradient-penalty?module=deep-learning-fundamentals"}>{"Spectral Normalization & Gradient Penalty"}</a>{". It asks how to control the sensitivity of learned functions. It is a new optimization/regularization question, not a continuation of the Gibbs sampler. Later "}<a href={"/learn/path/full-curriculum/modern-hopfield-networks?module=deep-learning-fundamentals"}>{"Modern Hopfield Networks"}</a>{" returns to energy and memory with a different state-update mechanism."}</Prose></section>
</div>};
