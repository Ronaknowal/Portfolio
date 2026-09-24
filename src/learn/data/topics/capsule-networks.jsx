// Complete revision-3 prepared manuscript; statically rendered at authoring time.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { CapsuleGrouping, CapsuleVoteLab, CapsuleSquashLab, CapsuleEvidenceLab, CapsuleFrozenInvestigation, CapsuleGeometryFigure, CapsuleEMLab, CapsuleClassicShape, CapsuleProgram, capsuleAsset } from '../../components/lesson-labs/CapsuleLabs.jsx';
export default {
 title: 'Capsule Networks',
 readTime: '~70 min read + code, live investigations and practice; optional deeper mechanics',
 hasIntegratedGuide: true,
 content: () => <div className="neural-lesson capsule-lesson"><LessonIntro prerequisites="Vector addition, matrix multiplication, convolution shapes and the idea of learning through a loss. Capsule-specific vocabulary and axes are introduced here." sections={[["1-a-capsule-is-a-bundle-of-properties","1. A capsule is a bundle of properties"],["2-from-a-part-to-a-prediction-about-a-whole","2. From a part to a prediction about a whole"],["3-routing-is-a-short-inference-computation","3. Routing is a short inference computation"],["4-build-the-classifier-and-its-objective","4. Build the classifier and its objective"],["5-run-a-controlled-experiment-on-real-digits","5. Run a controlled experiment on real digits"],["6-geometry-reconstruction-and-a-useful-failure","6. Geometry, reconstruction and a useful failure"],["7-deeper-mechanics-gradients-and-explicit-coordinate-frames","7. Deeper mechanics: gradients and explicit coordinate frames"],["follow-routing-all-the-way-into-a-trainable-program","Follow routing all the way into a trainable program"],["8-architecture-costs-and-alternative-routing-designs","8. Architecture costs and alternative routing designs"],["9-practice-implement-explain-and-compare","9. Practice: implement, explain and compare"],["10-continue-and-learn-another-way","10. Continue and learn another way"]]}>Follow one image from grouped properties to votes, assignments and class vectors. Build the mechanism, train a controlled model, and test what its geometry actually supports.</LessonIntro>
<Prose>{""}<strong>{"Explore as you read."}</strong>{" Edit capsule votes, routing iterations, vector magnitude/direction and supported retained image/latent coordinates. Follow the coupling rows, vote contributions, squash length/direction, current parent vectors and saved/frozen-model outputs. Step routing to inspect its computation, with all current outputs visible. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to distinguish agreement from activation magnitude, pose changes from class evidence and a model intervention from a new empirical result."}</Prose>

<Prose>{"A wheel detector firing twice is not enough to recognize a bicycle. The wheels also need a plausible arrangement relative to a frame. A "}<strong>{"capsule network"}</strong>{" tries to combine evidence about a part's presence with a vector or matrix describing its properties, then asks whether several parts inspect a compatible whole."}</Prose>

<Prose>{"In "}<a href={"/learn/path/full-curriculum/convnext-modern-cnn-designs?module=deep-learning-fundamentals"}>{"ConvNeXt"}</a>{", we changed how a convolutional network mixes spatial and channel information. Here the question changes: "}<strong>{"can the network decide, for this particular input, which higher-level entity should receive each part's evidence?"}</strong>{" We will build that computation, train a small classifier and test what the computation does and does not establish."}</Prose>

<Prose>{""}<strong>{"First pass:"}</strong>{" follow sections 1–6, run the small offline program or inspect its saved results, then attempt practice 1–5 and 8. You only need vector addition, matrix multiplication and the idea that a loss guides parameter updates. Sections 7–8 and practice 6–7 develop derivatives, matrix routing and engineering tradeoffs; they are a deeper branch, not a condition for understanding the core lesson."}</Prose>

<H2>{"1. A capsule is a bundle of properties"}</H2>

<Prose>{"Imagine a local image feature represented by the vector "}<InlineMath>{"u=(0.3,0.4)"}</InlineMath>{". Its length is "}<InlineMath>{"0.5"}</InlineMath>{". In a vector-capsule design, length is used as a presence "}<strong>{"score"}</strong>{" and the remaining variation can carry information useful for describing the feature."}</Prose>

<Prose>{"This does not mean coordinate 1 must be “rotation” and coordinate 2 must be “width.” A network can learn mixed, entangled coordinates. Calling a vector a pose vector is an architectural intention; interpreting a coordinate physically requires evidence from controlled input changes or reconstruction experiments."}</Prose>

<Prose>{"A usual CNN feature tensor already contains multiple channels at multiple positions. Capsules make a particular grouping and downstream computation explicit:"}</Prose>

<NeuralTable caption={"1. A capsule is a bundle of properties"} headers={[<>{"Representation"}</>,<>{"What is stored at one location?"}</>,<>{"How is it combined later?"}</>]} rows={[[<>{"Ordinary feature map"}</>,<>{"Several scalar channel values"}</>,<>{"Fixed learned convolutions or other mixing"}</>],[<>{"Vector capsule"}</>,<>{"A group of coordinates, such as an 8-vector"}</>,<>{"Transform into candidate whole-vectors, then combine by routing"}</>],[<>{"Matrix capsule"}</>,<>{"A pose matrix plus a separate activation scalar"}</>,<>{"Transform into candidate matrices, then estimate agreement and activation"}</>]]} />

<Prose>{"The distinction is not “CNNs contain no geometry.” Convolutions retain a spatial grid; channels can encode positional or orientation-sensitive information. Pooling can discard some exact detail, but its effect depends on the operation, boundaries and task. Capsule systems commonly begin with ordinary convolutions."}</Prose>

<Prose>{"A useful mental picture is an "}<strong>{"arrow"}</strong>{", not a glowing neuron: arrow direction carries a multidimensional state and arrow length carries a bounded score. The zero arrow has no direction. Ten class-capsule lengths need not sum to one, so they are not a softmax distribution or automatically calibrated probabilities."}</Prose>

<CapsuleGrouping />

<H2>{"2. From a part to a prediction about a whole"}</H2>

<Prose>{"A front wheel and a back wheel should make different predictions about the bicycle's center. The relation between each part and the whole matters."}</Prose>

<Prose>{"For child capsule "}<InlineMath>{"i"}</InlineMath>{" and candidate parent "}<InlineMath>{"j"}</InlineMath>{", learn a transformation matrix "}<InlineMath>{"W_{ij}"}</InlineMath>{". The "}<strong>{"vote"}</strong>{""}</Prose>

<div className="neural-equation"><MathBlock>{"\\widehat u_{j|i}=W_{ij}u_i"}</MathBlock></div>

<Prose>{"is child "}<InlineMath>{"i"}</InlineMath>{"'s prediction of parent "}<InlineMath>{"j"}</InlineMath>{"'s representation. If "}<InlineMath>{"u_i"}</InlineMath>{" has 4 coordinates and the parent has 8, "}<InlineMath>{"W_{ij}"}</InlineMath>{" has shape "}<InlineMath>{"8\\times4"}</InlineMath>{". A child has a different vote for each parent. Comparing its untransformed vector directly with every parent would skip the learned relationship."}</Prose>

<Prose>{"Suppose three children send these two-dimensional votes:"}</Prose>

<NeuralTable caption={"2. From a part to a prediction about a whole"} headers={[<>{"Child"}</>,<>{"Vote for parent A"}</>,<>{"Vote for parent B"}</>]} rows={[[<>{"1"}</>,<>{""}<InlineMath>{"(2,0)"}</InlineMath>{""}</>,<>{""}<InlineMath>{"(0,1)"}</InlineMath>{""}</>],[<>{"2"}</>,<>{""}<InlineMath>{"(2,0)"}</InlineMath>{""}</>,<>{""}<InlineMath>{"(0,-1)"}</InlineMath>{""}</>],[<>{"3"}</>,<>{""}<InlineMath>{"(0,1)"}</InlineMath>{""}</>,<>{""}<InlineMath>{"(0,2)"}</InlineMath>{""}</>]]} />

<Prose>{"This is a constructed arithmetic example, not measured image features. Children 1 and 2 reinforce each other for A but oppose each other for B. Child 3 could support B."}</Prose>

<Prose>{"We need a way to combine the votes without fixing every connection strength for every image. Define "}<InlineMath>{"c_{ij}"}</InlineMath>{" as child "}<InlineMath>{"i"}</InlineMath>{"'s fraction assigned to parent "}<InlineMath>{"j"}</InlineMath>{". Initially, with two parents, each row is "}<InlineMath>{"(0.5,0.5)"}</InlineMath>{"."}</Prose>

<Prose>{"The tentative parent inputs are "}<strong>{"weighted sums"}</strong>{":"}</Prose>

<div className="neural-equation"><MathBlock>{"s_j=\\sum_i c_{ij}\\widehat u_{j|i}."}</MathBlock></div>

<Prose>{"Initially "}<InlineMath>{"s_A=(2,0.5)"}</InlineMath>{" and "}<InlineMath>{"s_B=(0,1)"}</InlineMath>{". The weights sum to one "}<strong>{"across parents for each child"}</strong>{". They generally do not sum to one across children for a parent. Therefore this operation is not a weighted average of the incoming votes."}</Prose>

<Prose>{"That distinction matters: duplicating two agreeing children can strengthen the parent input. A routing diagram should show both the incoming arrows and their scalar contribution weights, not only a heatmap."}</Prose>

<H3>{"Keeping the output length below one"}</H3>

<Prose>{"Use the squash function"}</Prose>

<div className="neural-equation"><MathBlock>{"v=\\operatorname{squash}(s)=\\frac{r}{1+r^2}s,\\qquad r=\\|s\\|."}</MathBlock></div>

<Prose>{"Its output length is"}</Prose>

<div className="neural-equation"><MathBlock>{"\\|v\\|=\\frac{r^2}{1+r^2}."}</MathBlock></div>

<Prose>{"Small inputs become very short; large inputs approach length one. Direction is preserved whenever "}<InlineMath>{"s\\ne0"}</InlineMath>{", and "}<InlineMath>{"\\operatorname{squash}(0)=0"}</InlineMath>{". This form avoids an explicit division by the norm at zero."}</Prose>

<Prose>{"For parent B, "}<InlineMath>{"r=1"}</InlineMath>{", so "}<InlineMath>{"v_B=(0,0.5)"}</InlineMath>{". For A, "}<InlineMath>{"r=\\sqrt{4.25}"}</InlineMath>{", giving "}<InlineMath>{"v_A\\approx(0.78535,0.19634)"}</InlineMath>{", length "}<InlineMath>{"0.80952"}</InlineMath>{". A has the stronger initial score."}</Prose>

<Prose>{"Do not read the output as “an 80.95% probability of a bicycle.” A bounded range and a probabilistic interpretation are different requirements."}</Prose>

<CapsuleSquashLab />

<H2>{"3. Routing is a short inference computation"}</H2>

<Prose>{""}<strong>{"Routing by agreement"}</strong>{" repeatedly revises the connection fractions within one forward pass. This is different from updating model parameters across training examples."}</Prose>

<Prose>{"Start a logit "}<InlineMath>{"b_{ij}=0"}</InlineMath>{" for each child–parent pair. A logit is an unconstrained score used by softmax. For each routing step:"}</Prose>

<ol><li>{"Compute "}<InlineMath>{"c_{ij}=\\exp(b_{ij})/\\sum_k\\exp(b_{ik})"}</InlineMath>{", normalizing over candidate parents."}</li><li>{"Sum the weighted votes into "}<InlineMath>{"s_j"}</InlineMath>{", then squash to obtain "}<InlineMath>{"v_j"}</InlineMath>{"."}</li><li>{"If another routing step remains, update "}<InlineMath>{"b_{ij}\\leftarrow b_{ij}+\\widehat u_{j|i}^{\\mathsf T}v_j"}</InlineMath>{"."}</li></ol>

<Prose>{"Subtracting the row maximum before exponentiation gives the same softmax with better numerical stability. The agreement is a "}<strong>{"dot product"}</strong>{": both direction and magnitude affect it. It is not cosine similarity unless the operands are explicitly normalized, which would define a different routing rule."}</Prose>

<Prose>{"For our first step, child 1's agreements are approximately "}<InlineMath>{"1.5707"}</InlineMath>{" with A and "}<InlineMath>{"0.5"}</InlineMath>{" with B. Child 2 agrees by "}<InlineMath>{"1.5707"}</InlineMath>{" with A and "}<InlineMath>{"-0.5"}</InlineMath>{" with B. Child 3 agrees by "}<InlineMath>{"0.1963"}</InlineMath>{" with A and "}<InlineMath>{"1.0"}</InlineMath>{" with B. Those differences change the next softmax rows."}</Prose>

<NeuralTable caption={"3. Routing is a short inference computation"} headers={[<>{"Step"}</>,<>{"Child 1 → A"}</>,<>{"Child 2 → A"}</>,<>{"Child 3 → A"}</>,<>{"A length"}</>,<>{"B length"}</>]} rows={[[<>{"1"}</>,<>{".5000"}</>,<>{".5000"}</>,<>{".5000"}</>,<>{".8095"}</>,<>{".5000"}</>],[<>{"2"}</>,<>{".7447"}</>,<>{".8880"}</>,<>{".3092"}</>,<>{".9150"}</>,<>{".6993"}</>],[<>{"3"}</>,<>{".8996"}</>,<>{".9900"}</>,<>{".1076"}</>,<>{".9346"}</>,<>{".7786"}</>]]} />

<Prose>{"Each B fraction is one minus the corresponding A fraction. Both parents can acquire substantial scores because different children support them."}</Prose>

<CapsuleVoteLab />

<details>

<summary>Worked changed-vote calculation</summary>

<Prose>{"The computed lengths become approximately .6491 for A and .8585 for B. Explain the cancellation, then explain why child 3's reassignment also matters. Try a different edited vote without being given its answer in advance."}</Prose>

</details>

<Prose>{"Two useful null cases guard against overinterpreting the animation:"}</Prose>

<ul><li>{"If every vote is zero, every output stays zero and every coupling stays uniform."}</li><li>{"If every child sends exactly the same vote to both parents, symmetry keeps the two parent outputs and each row's fractions equal. Routing cannot invent evidence to break this symmetry."}</li></ul>

<Prose>{"Repeated agreement often makes rows increasingly concentrated. It need not change the predicted class, converge to the correct grouping or improve generalization. “Run until certain” is not a valid stopping rule. The number of iterations is part of the model configuration."}</Prose>

<H3>{"Where does learning happen?"}</H3>

<Prose>{"The convolution weights, "}<InlineMath>{"W_{ij}"}</InlineMath>{" and decoder weights are persistent learned parameters. Votes, logits, couplings and parent vectors are input-dependent intermediate values. Our logits restart at zero for each new image and every forward pass; they are not carried from the preceding image."}</Prose>

<Prose>{"A finite sequence of matrix products, softmax, sums and squash operations can be differentiated. Backpropagation can flow through all routing steps. Some implementations detach intermediate routing computations; that preserves the forward numbers for a fixed input but changes the gradient used to learn the parameters. It is a deliberate algorithmic choice, not a requirement imposed by routing."}</Prose>

<Prose>{"The distinction will return in "}<a href={"/learn/path/full-curriculum/rnns-lstms-grus?module=deep-learning-fundamentals"}>{"RNNs, LSTMs and GRUs"}</a>{": repetition inside routing refines assignments for one image, whereas recurrence along a sequence updates state as new observations arrive."}</Prose>

<H2>{"4. Build the classifier and its objective"}</H2>

<Prose>{"Our small model receives one real "}<InlineMath>{"8\\times8"}</InlineMath>{" grayscale digit. Here is the complete shape path; "}<InlineMath>{"B"}</InlineMath>{" means batch size."}</Prose>

<NeuralTable caption={"4. Build the classifier and its objective"} headers={[<>{"Stage"}</>,<>{"Output shape"}</>,<>{"Meaning"}</>]} rows={[[<>{"Image"}</>,<>{""}<InlineMath>{"B\\times1\\times8\\times8"}</InlineMath>{""}</>,<>{"Pixel intensities divided by 16"}</>],[<>{"Conv "}<InlineMath>{"3\\times3"}</InlineMath>{", padding 1; ReLU"}</>,<>{""}<InlineMath>{"B\\times32\\times8\\times8"}</InlineMath>{""}</>,<>{"Local scalar features"}</>],[<>{"Conv "}<InlineMath>{"3\\times3"}</InlineMath>{", stride 2, padding 1"}</>,<>{""}<InlineMath>{"B\\times16\\times4\\times4"}</InlineMath>{""}</>,<>{"Four capsule types, four coordinates each"}</>],[<>{"Regroup and squash"}</>,<>{""}<InlineMath>{"B\\times64\\times4"}</InlineMath>{""}</>,<>{""}<InlineMath>{"4\\cdot4"}</InlineMath>{" locations × 4 types"}</>],[<>{"Learned votes"}</>,<>{""}<InlineMath>{"B\\times64\\times10\\times8"}</InlineMath>{""}</>,<>{"Every child predicts all ten digit classes"}</>],[<>{"Routing and squash"}</>,<>{""}<InlineMath>{"B\\times10\\times8"}</InlineMath>{""}</>,<>{"One vector per class"}</>],[<>{"Vector lengths and argmax"}</>,<>{""}<InlineMath>{"B\\times10"}</InlineMath>{", then "}<InlineMath>{"B"}</InlineMath>{""}</>,<>{"Scores, then predicted digit"}</>]]} />

<Prose>{"Regrouping must preserve the coordinate group. In the supplied program, the primary channels are ordered by capsule type, then coordinate. We reshape to "}<InlineMath>{"B\\times4_{\\text{type}}\\times4_{\\text{coord}}\\times4_H\\times4_W"}</InlineMath>{", permute to location–type–coordinate order, then flatten the child index. Flattening the original tensor indiscriminately could group coordinates from different locations."}</Prose>

<Prose>{"For a true class "}<InlineMath>{"k"}</InlineMath>{", the margin objective encourages its length to reach at least .9 and other lengths to stay at most .1:"}</Prose>

<div className="neural-equation"><MathBlock>{"L_{\\text{margin}}\n=\\sum_j\\left[T_j\\max(0,.9-\\|v_j\\|)^2\n+.5(1-T_j)\\max(0,\\|v_j\\|-.1)^2\\right],"}</MathBlock></div>

<Prose>{"where "}<InlineMath>{"T_j=1"}</InlineMath>{" for the actual class and 0 otherwise. Average this sum over images. If the true-class length is .7 and one wrong-class length is .3, with all other wrong lengths at most .1, the loss is "}<InlineMath>{".2^2+.5(.2^2)=.06"}</InlineMath>{"."}</Prose>

<Prose>{"The loss has a zero-penalty region, rather than continually pushing the true length to one. The factor .5 changes the cost of wrong-class activation. This is not cross-entropy, so do not feed the lengths into a cross-entropy API as though they were unconstrained logits."}</Prose>

<H3>{"Reconstruction asks the vector to retain useful detail"}</H3>

<Prose>{"Mask all class vectors except one, flatten the ten 8-vectors into 80 values, and decode through "}<InlineMath>{"80\\to64\\to128\\to64"}</InlineMath>{", using ReLU between layers and sigmoid on the final pixels. During training, select the "}<strong>{"true"}</strong>{" class vector for the auxiliary reconstruction objective:"}</Prose>

<div className="neural-equation"><MathBlock>{"L=L_{\\text{margin}}+.0005\\sum_{p=1}^{64}(\\widehat x_p-x_p)^2."}</MathBlock></div>

<Prose>{"The squared errors are summed over pixels and averaged over images. On 64 pixels, this is equivalent to adding "}<InlineMath>{".032"}</InlineMath>{" times pixel-mean MSE. Accidentally using mean MSE with coefficient .0005 makes this term 64 times smaller."}</Prose>

<Prose>{"The decoder is encouraged to retain image detail, but a reconstruction objective does not identify which latent axis must represent which physical factor. At classification time no label is needed: predict with the longest vector. For ordinary reconstruction at inference, mask using that predicted class. A reconstruction conditioned on the known true class is a separate diagnostic with extra information; we report it separately."}</Prose>

<Prose>{"This design has one class capsule for each digit. It cannot separately represent two different instances of the same digit in those ten slots. Representing an image containing two 3s would require additional instance capacity and an appropriate objective; selecting two different class slots does not solve that case."}</Prose>

<H2>{"5. Run a controlled experiment on real digits"}</H2>

<Prose>{"The offline "}<a href={"/learn-assets/capsule-networks/digits-400.csv"}>{"400-image CSV"}</a>{" contains optical handwritten digits from the "}<a href={"https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits"}>{"UCI dataset"}</a>{", attributed to E. Alpaydin and C. Kaynak and distributed under CC BY 4.0. Each row has 64 integer intensities in 0–16 and a digit label. This is not MNIST."}</Prose>

<Prose>{"We take 40 images per class and make a fixed stratified subdivision: 280 training images and 120 development images, using seed 22. The program checks distinct source IDs and complete pixel vectors before splitting. Writer identifiers are absent, so this does not establish performance on independent writers. The development set is used for the comparisons shown here; there is no untouched final test or deployment claim."}</Prose>

<Prose>{"Our baseline is the "}<strong>{"same capsule model with one routing step"}</strong>{". Since every coupling is initially "}<InlineMath>{"1/10"}</InlineMath>{", it is a uniform-routing classifier. Compare it with a separately trained three-step model, keeping architecture, initial common weights, minibatch draws, optimizer and update count paired. This isolates an actionable routing choice more directly than comparing unrelated large CNN and capsule systems."}</Prose>

<Prose>{"Save "}<a href={"/learn-assets/capsule-networks/capsule-learning.py"}>{"capsule-learning.py"}</a>{" beside the CSV. The complete program includes model definitions, input validation, split, training, evaluation and JSON export. It does not download pretrained weights."}</Prose>

<CapsuleProgram /><Prose>The complete source below is deferred until you open it. The CSV, import helper and mechanics program are available beside it; save these linked files in one directory for the documented local commands.</Prose><p><a href={capsuleAsset + "native-verification.json"}>Actual execution, environment and numerical checks</a> · <a href={capsuleAsset + "calculated-inputs.json"}>Complete measured record</a></p>

<CodeBlock language={"bash"}>{"python -m pip install numpy==2.3.5 scikit-learn==1.9.1 torch==2.14.0\npython capsule-learning.py"}</CodeBlock>

<Prose>{"The author run used Python 3.12.14 and PyTorch 2.14.0+cpu, one CPU thread. Choose the CPU package source appropriate to your platform if the general package command offers a different accelerator build. Numerical results can vary with platform or future dependency changes; the saved arrays give an exact reference for this run."}</Prose>

<Prose>{"There are six fits: seeds 1, 2 and 3, each with one or three routing steps. Every fit performs 600 Adam updates, learning rate .003, batch size 64 sampled with replacement. The vote matrices start from a normal distribution with standard deviation .1; other layers use PyTorch's defaults. No augmentation, early stopping or checkpoint selection is used. Both variants contain 47,184 parameters, including the decoder."}</Prose>

<Prose>{"The central routing function is short enough to inspect. Votes have shape "}<InlineMath>{"B,I,J,D"}</InlineMath>{"; the parent axis is 2:"}</Prose>

<CodeBlock language={"python"}>{"def squash(vectors):\n    radius = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)\n    return vectors * radius / (1 + radius.square())\n\ndef route(votes, iterations=3):\n    logits = votes.new_zeros(votes.shape[:-1])\n    for step in range(iterations):\n        coupling = logits.softmax(dim=2)\n        sums = (coupling[..., None] * votes).sum(dim=1)\n        output = squash(sums)\n        if step < iterations - 1:\n            logits = logits + (votes * output[:, None]).sum(dim=-1)\n    return output"}</CodeBlock>

<Prose>{"The downloaded program adds optional trace capture and an explicitly selected stop-gradient demonstration; training uses the full derivative. The displayed function is the same default computation, with those teaching options removed for readability."}</Prose>

<H3>{"What actually happened?"}</H3>

<Prose>{"All six models classified all 280 training images correctly at the final update. Development performance was:"}</Prose>

<NeuralTable caption={"What actually happened?"} headers={[<>{"Seed"}</>,<>{"Routing during training and evaluation"}</>,<>{"Correct / 120"}</>,<>{"Margin loss"}</>,<>{"Reconstruction MSE using predicted class"}</>]} rows={[[<>{"1"}</>,<>{"1"}</>,<>{"117"}</>,<>{".04680"}</>,<>{".03126"}</>],[<>{"1"}</>,<>{"3"}</>,<>{"117"}</>,<>{".02364"}</>,<>{".03334"}</>],[<>{"2"}</>,<>{"1"}</>,<>{"117"}</>,<>{".04823"}</>,<>{".03103"}</>],[<>{"2"}</>,<>{"3"}</>,<>{"116"}</>,<>{".03311"}</>,<>{".03401"}</>],[<>{"3"}</>,<>{"1"}</>,<>{"117"}</>,<>{".04608"}</>,<>{".03325"}</>],[<>{"3"}</>,<>{"3"}</>,<>{"118"}</>,<>{".02497"}</>,<>{".03552"}</>]]} />

<Prose>{"The three-step model's margin loss is lower in every paired run, but classification ties, loses one example or gains one example. Stronger margins do not necessarily change the largest score. The reconstruction term is not a proxy for classification quality either."}</Prose>

<Prose>{"A training-mean-image baseline has development pixel MSE .07296. Both decoders improve on that crude reconstruction baseline. Their predicted-mask reconstructions differ from their true-label-conditioned reconstructions: for seed 1, the corresponding MSEs are .03126 versus .03041 for one-step routing and .03334 versus .03293 for three steps. The lower diagnostic error uses information unavailable at ordinary inference."}</Prose>

<CapsuleEvidenceLab />

<H3>{"A different question: change routing after fitting"}</H3>

<Prose>{"Hold each trained model fixed and evaluate it with different routing counts:"}</Prose>

<NeuralTable caption={"A different question: change routing after fitting"} headers={[<>{"Seed"}</>,<>{"Steps used to train"}</>,<>{"Evaluate with 1"}</>,<>{"With 2"}</>,<>{"With 3"}</>,<>{"With 5"}</>]} rows={[[<>{"1"}</>,<>{"1"}</>,<>{"117"}</>,<>{"117"}</>,<>{"117"}</>,<>{"116"}</>],[<>{"1"}</>,<>{"3"}</>,<>{"118"}</>,<>{"118"}</>,<>{"117"}</>,<>{"117"}</>],[<>{"2"}</>,<>{"1"}</>,<>{"117"}</>,<>{"117"}</>,<>{"117"}</>,<>{"116"}</>],[<>{"2"}</>,<>{"3"}</>,<>{"118"}</>,<>{"117"}</>,<>{"116"}</>,<>{"116"}</>],[<>{"3"}</>,<>{"1"}</>,<>{"117"}</>,<>{"117"}</>,<>{"117"}</>,<>{"118"}</>],[<>{"3"}</>,<>{"3"}</>,<>{"117"}</>,<>{"118"}</>,<>{"118"}</>,<>{"118"}</>]]} />

<Prose>{"All entries are correct counts out of the same 120 development images. Equal counts need not mean identical predictions: the seed-3 one-step model changes one prediction at inference step 2, although its correct count stays 117."}</Prose>

<Prose>{"This intervention tests sensitivity of "}<strong>{"fixed weights"}</strong>{". It is different from training the model under a new routing configuration. Choosing a preferred inference count after seeing this table consumes the development comparison; it is not an unbiased final evaluation of a new setting."}</Prose>

<Prose>{"The result supports a bounded conclusion: three-step routing was not consistently better for this small experiment. It does not establish that routing never helps. A published controlled investigation similarly emphasizes testing routing against uniform alternatives, but uses different architectures, datasets and procedures. "}<a href={"https://proceedings.mlr.press/v101/paik19a.html"}>{"Paik, Kwak and Kim, ACML 2019"}</a>{""}</Prose>

<H2>{"6. Geometry, reconstruction and a useful failure"}</H2>

<Prose>{"A system can recognize an object despite movement without retaining a predictable geometric representation. Conversely, a representation can move predictably while the final classifier still makes errors."}</Prose>

<ul><li>{""}<strong>{"Invariance:"}</strong>{" "}<InlineMath>{"f(Tx)=f(x)"}</InlineMath>{". The chosen output does not change under transformation "}<InlineMath>{"T"}</InlineMath>{"."}</li><li>{""}<strong>{"Equivariance:"}</strong>{" "}<InlineMath>{"f(Tx)=\\rho(T)f(x)"}</InlineMath>{". The output changes according to a specified corresponding transformation "}<InlineMath>{"\\rho(T)"}</InlineMath>{"."}</li></ul>

<Prose>{"For an image classification label, invariance may be desirable for a small translation that preserves the digit. For a position estimate, translation equivariance is usually necessary: the estimated location should move. A rotation can change the semantic label in some tasks, so desired invariances must come from the task contract."}</Prose>

<Prose>{"A learned capsule vector does not by itself define "}<InlineMath>{"\\rho(T)"}</InlineMath>{". Without that definition and a check of the equality, a higher transformed-image accuracy is evidence about robustness, not a proof of equivariance."}</Prose>

<H3>{"A shift test with no retraining"}</H3>

<Prose>{"Shift every development image one pixel right or down, filling the exposed boundary with zero and discarding pixels that leave the frame. Evaluate with the routing count used for training."}</Prose>

<NeuralTable caption={"A shift test with no retraining"} headers={[<>{"Seed"}</>,<>{"Trained steps"}</>,<>{"Unchanged"}</>,<>{"One pixel right"}</>,<>{"One pixel down"}</>]} rows={[[<>{"1"}</>,<>{"1"}</>,<>{"117"}</>,<>{"69"}</>,<>{"75"}</>],[<>{"1"}</>,<>{"3"}</>,<>{"117"}</>,<>{"61"}</>,<>{"72"}</>],[<>{"2"}</>,<>{"1"}</>,<>{"117"}</>,<>{"66"}</>,<>{"72"}</>],[<>{"2"}</>,<>{"3"}</>,<>{"116"}</>,<>{"60"}</>,<>{"77"}</>],[<>{"3"}</>,<>{"1"}</>,<>{"117"}</>,<>{"72"}</>,<>{"76"}</>],[<>{"3"}</>,<>{"3"}</>,<>{"118"}</>,<>{"65"}</>,<>{"78"}</>]]} />

<Prose>{"The model is highly sensitive to these shifts. The experiment uses small images, stride 2, location-specific vote matrices and no shift augmentation. Cropping can also remove meaningful strokes; inspect individual transformed inputs rather than assuming every transformation perfectly preserves the label. Zero shift reproduces the unchanged predictions."}</Prose>

<Prose>{"This is a useful failure. Routing's internal agreement does not manufacture the missing coverage of transformed data or constrain every preceding layer to respect a symmetry."}</Prose>

<CapsuleFrozenInvestigation />

<H3>{"What can a latent-coordinate experiment tell us?"}</H3>

<Prose>{"Choose one class vector, keep the decoder and mask fixed, and change one coordinate by "}<InlineMath>{"-.1,0,+.1"}</InlineMath>{". Display the three reconstructions at the same intensity scale. This probes how the "}<strong>{"learned decoder"}</strong>{" responds to that coordinate near that specimen."}</Prose>

<Prose>{"It may alter several image properties at once. Even if a change resembles thickness in one image, call it a local observed effect until it recurs under broader controlled tests. A coordinate edit is not guaranteed to lie on the distribution of vectors the encoder actually produces."}</Prose>

<Prose>{"There is also a precise null case: if the decoder mask selects class 4, changing only class 5's vector cannot affect the masked decoder input. Changing a caption's label without changing the actual mask cannot affect any arithmetic."}</Prose>

<H3>{"An unusual application: separate overlapping objects"}</H3>

<Prose>{"Why reconstruct one class at a time? In an image containing two different digits, two class capsules can condition two reconstructions. The auxiliary task asks each selected representation to account for a different component rather than the combined image."}</Prose>

<Prose>{"This requires appropriate paired component targets and a multi-object objective; the single-digit model above was not trained for it. A careful experiment must build training composites from training specimens and held-out composites from held-out specimens. Millions of pairings of a smaller source set do not become millions of independent original specimens. The original capsule work explored this direction on overlapping digits. "}<a href={"https://arxiv.org/abs/1710.09829"}>{"Dynamic Routing Between Capsules"}</a>{""}</Prose>

<Prose>{"The same part-to-whole question can arise in video: frame-level or local motion evidence may support an action occurring over a region and interval. The representation, instance capacity and evaluation unit must then include time. This is a reason to investigate capsules, not evidence that this digit model already solves action localization. The "}<a href={"https://www.crcv.ucf.edu/cvpr2019-tutorial/"}>{"UCF CVPR tutorial"}</a>{" includes a separate video-capsule session for that application."}</Prose>

<H2>{"7. Deeper mechanics: gradients and explicit coordinate frames"}</H2>

<H3>{"Squash changes radial and sideways sensitivity differently"}</H3>

<Prose>{"Write "}<InlineMath>{"s=rq"}</InlineMath>{", where "}<InlineMath>{"q"}</InlineMath>{" is a unit vector. A small change parallel to "}<InlineMath>{"q"}</InlineMath>{" changes length; a perpendicular change initially changes direction. The squash Jacobian has eigenvalues"}</Prose>

<div className="neural-equation"><MathBlock>{"\\lambda_{\\text{radial}}=\\frac{2r}{(1+r^2)^2},\\qquad\n\\lambda_{\\text{tangent}}=\\frac{r}{1+r^2}."}</MathBlock></div>

<Prose>{"Both approach zero at the origin and at very large radius, at different rates. Consequently, making initial votes arbitrarily tiny or letting summed inputs become huge can reduce useful gradients. The radial output-length curve "}<InlineMath>{"r^2/(1+r^2)"}</InlineMath>{" has its inflection at "}<InlineMath>{"r=1/\\sqrt3"}</InlineMath>{", not at 1."}</Prose>

<Prose>{"For "}<InlineMath>{"s=(.3,.4)"}</InlineMath>{", "}<InlineMath>{"r=.5"}</InlineMath>{", radial sensitivity is .64 and tangent sensitivity .4. For "}<InlineMath>{"s=(3,4)"}</InlineMath>{", "}<InlineMath>{"r=5"}</InlineMath>{", they are approximately .01479 and .19231. A very long vector is much harder to lengthen than to rotate locally."}</Prose>

<Prose>{"The "}<a href={"/learn-assets/capsule-networks/capsule-mechanics.py"}>{"mechanics program"}</a>{" computes the analytical Jacobian and checks it by central differences. At zero the exact derivative is zero; a finite difference has a small step-dependent residual. This is a numerical approximation, not a contradictory derivative."}</Prose>

<Prose>{"For the three-step routing fixture, differentiating the full computation agrees with central differences to about "}<InlineMath>{"1.2\\times10^{-10}"}</InlineMath>{". Detaching earlier routing calculations gives the same scalar loss but a gradient differing by up to .03752. Both can be coded; only the full derivative matches the stated full forward function's derivative."}</Prose>

<H3>{"Why explicit matrices can help—and what they do not guarantee"}</H3>

<Prose>{"Suppose a part really has a homogeneous 2D coordinate frame"}</Prose>

<div className="neural-equation"><MathBlock>{"M_i=\\begin{bmatrix}1&0&2\\\\0&1&3\\\\0&0&1\\end{bmatrix}"}</MathBlock></div>

<Prose>{"and its relation to the whole is"}</Prose>

<div className="neural-equation"><MathBlock>{"W_{ij}=\\begin{bmatrix}1&0&-1\\\\0&1&0\\\\0&0&1\\end{bmatrix}."}</MathBlock></div>

<Prose>{"Then "}<InlineMath>{"M_iW_{ij}"}</InlineMath>{" predicts a whole located at "}<InlineMath>{"(1,3)"}</InlineMath>{". Rotate the entire scene by "}<InlineMath>{"90^\\circ"}</InlineMath>{", using"}</Prose>

<div className="neural-equation"><MathBlock>{"G=\\begin{bmatrix}0&-1&0\\\\1&0&0\\\\0&0&1\\end{bmatrix}."}</MathBlock></div>

<Prose>{"Associativity gives "}<InlineMath>{"(GM_i)W_{ij}=G(M_iW_{ij})"}</InlineMath>{", so the predicted whole moves to "}<InlineMath>{"(-3,1)"}</InlineMath>{" consistently. The part–whole relation stays fixed while the viewing frame changes."}</Prose>

<Prose>{"This is an exact calculation "}<strong>{"because"}</strong>{" these matrices have an explicitly supplied geometric meaning and transform by left multiplication. A learned encoder must still produce frames with the promised transformation behavior. A generic learned "}<InlineMath>{"4\\times4"}</InlineMath>{" array does not automatically become a rigid camera pose."}</Prose>

<CapsuleGeometryFigure />

<Prose>{"For vector votes, a linear map needs"}</Prose>

<div className="neural-equation"><MathBlock>{"W\\rho_{\\text{in}}(T)=\\rho_{\\text{out}}(T)W"}</MathBlock></div>

<Prose>{"to preserve a chosen symmetry. Arbitrary "}<InlineMath>{"W"}</InlineMath>{" need not satisfy it. For "}<InlineMath>{"W=\\operatorname{diag}(2,1)"}</InlineMath>{", "}<InlineMath>{"u=(1,2)"}</InlineMath>{" and a "}<InlineMath>{"90^\\circ"}</InlineMath>{" rotation "}<InlineMath>{"R"}</InlineMath>{", "}<InlineMath>{"WRu=(-4,1)"}</InlineMath>{" while "}<InlineMath>{"RWu=(-2,2)"}</InlineMath>{"."}</Prose>

<Prose>{"Squash itself commutes with an orthogonal rotation because rotation preserves the norm. It does not commute with arbitrary scaling: "}<InlineMath>{"\\operatorname{squash}(2u)\\ne2\\operatorname{squash}(u)"}</InlineMath>{". Exact symmetry is a whole-computation constraint, not a descriptive name. "}<a href={"https://arxiv.org/abs/1602.07576"}>{"Group Equivariant Convolutional Networks"}</a>{" develops architectures that explicitly constrain transformations."}</Prose>

<H3>{"Matrix capsules and EM routing"}</H3>

<Prose>{"Matrix capsules separate an activation scalar "}<InlineMath>{"a_i"}</InlineMath>{" from pose matrix "}<InlineMath>{"M_i"}</InlineMath>{". Votes are "}<InlineMath>{"V_{ij}=M_iW_{ij}"}</InlineMath>{". Flatten the entries of a vote into coordinates "}<InlineMath>{"h"}</InlineMath>{" only for the statistics."}</Prose>

<Prose>{"Instead of repeatedly adding dot-product agreement, a diagonal-Gaussian routing procedure estimates a mean and variance for the votes supporting each parent. Let "}<InlineMath>{"R_{ij}"}</InlineMath>{" be a child's normalized responsibility and "}<InlineMath>{"q_{ij}=a_iR_{ij}"}</InlineMath>{". Then"}</Prose>

<div className="neural-equation"><MathBlock>{"n_j=\\sum_iq_{ij},\\quad\n\\mu_{jh}=\\frac{\\sum_iq_{ij}V_{ijh}}{n_j},\\quad\n\\sigma^2_{jh}=\\frac{\\sum_iq_{ij}(V_{ijh}-\\mu_{jh})^2}{n_j}."}</MathBlock></div>

<Prose>{"The child activation scales its contribution. Each iteration recomputes "}<InlineMath>{"q=aR"}</InlineMath>{"; repeatedly multiplying a previously scaled "}<InlineMath>{"q"}</InlineMath>{" by "}<InlineMath>{"a"}</InlineMath>{" would incorrectly suppress children again and again. A parent with zero effective mass has no identified mean. A numerical denominator guard prevents a crash but does not create evidence; report the no-evidence case. A variance floor similarly prevents singular densities while changing the fitted spread."}</Prose>

<Prose>{"Estimate a parent activation from a coding-cost expression, then revise responsibilities using a Gaussian log density plus log parent activation, normalized over parents. Our complete constructed demonstration uses"}</Prose>

<div className="neural-equation"><MathBlock>{"\\mathrm{cost}_j=\\sum_h n_j(\\beta_u+\\log\\sigma_{jh}),\\quad\na_j=\\operatorname{sigmoid}\\{\\lambda(\\beta_a-\\mathrm{cost}_j)\\},"}</MathBlock></div>

<div className="neural-equation"><MathBlock>{"R_{ij}=\\operatorname{softmax}_j\\left[\n\\log a_j-\\frac12\\sum_h\\left(\\log(2\\pi\\sigma^2_{jh})\n+\\frac{(V_{ijh}-\\mu_{jh})^2}{\\sigma^2_{jh}}\\right)\\right]."}</MathBlock></div>

<Prose>{"It fixes "}<InlineMath>{"\\beta_u=\\beta_a=0"}</InlineMath>{", variance floor .01 and inverse temperatures .5, .75 and 1. These are declared illustration settings; a trained matrix-capsule model learns cost parameters and uses a chosen schedule."}</Prose>

<Prose>{"For three children with activations "}<InlineMath>{"1,1,.5"}</InlineMath>{", parent-A first coordinates "}<InlineMath>{"0,.2,2"}</InlineMath>{" and parent-B first coordinates "}<InlineMath>{"0,3,3.2"}</InlineMath>{", uniform responsibilities give mass 1.25 per parent. The first means are .48 and 1.84. After three rounds they are approximately .10881 and 2.81927. The second coordinate is zero for every vote, so its variance hits the stated floor. Making the third child inactive makes edits to its votes irrelevant to the estimated means."}</Prose>

<Prose>{"The resemblance to "}<a href={"/learn/path/full-curriculum/gaussian-mixture-models-gmm-em-algorithm?module=classical-ml"}>{"Gaussian-mixture EM"}</a>{" is useful, but each capsule parent sees a differently transformed version of the children, and parent activations do not sum to one. It is not ordinary maximum-likelihood fitting of one common observed dataset. The matrix-capsule paper discusses the change-of-variables issue when comparing densities in different transformed spaces. "}<a href={"https://www.cs.toronto.edu/~hinton/absps/EMcapsules.pdf"}>{"Matrix Capsules with EM Routing"}</a>{""}</Prose>

<CapsuleEMLab /><CapsuleProgram file="capsule-mechanics.py" title="Read the complete NumPy routing, derivative, geometry and EM program" />

<H2>{"Follow routing all the way into a trainable program"}</H2>

<Prose>{"The runnable scratch route is split by purpose, not by missing work. "}<a href={"/learn-assets/capsule-networks/capsule-mechanics.py"}>{"capsule-mechanics.py"}</a>{" owns NumPy votes, stable softmax, squash, routing iterations and the bounded diagonal-EM illustration. "}<a href={"/learn-assets/capsule-networks/capsule-learning.py"}>{"capsule-learning.py"}</a>{" owns the differentiable Torch routing, "}<code>{"TinyCapsules"}</code>{", margin loss, reconstruction and complete fit. The Torch tensor implementation is the ordinary research route: there is no universal standard capsule layer whose import can replace specifying the routing algorithm. Reusing "}<code>{"einsum"}</code>{", linear layers and autograd does not hide routing; the code explicitly updates its coupling logits and sums weighted votes."}</Prose>

<Prose>{"The paired route compares the same votes and iteration count, not two independently fitted classifiers. "}<code>{"author-checks.py"}</code>{" reconstructs saved encoder/routing/reconstruction outputs through a separate NumPy path. The lesson's gradient branch explains how gradients flow through the iterative computation. Detaching intermediate agreements would change that training algorithm even if its forward result stayed identical. Softmax runs over candidate "}<strong>{"parents for each child"}</strong>{"; moving that axis changes who competes for responsibility."}</Prose>

<CapsuleProgram file="author-checks.py" title="Read the independent NumPy encoder and saved-state comparison" />

<Prose>{"Dynamic routing with B examples, I child capsules, J parents, vote width D and R rounds uses O(BIJD·R) routing arithmetic and O(BIJD) vote storage, apart from the learned vote transforms. Parent-batched contractions are appropriate for the small inspected classifier; manufacturing an extra all-pairs child tensor is not. Diagonal EM has a different Gaussian/statistical meaning and stays explicitly a small illustrative calculation, not an implementation claim for the complete Matrix Capsules paper or its convolutional pose system."}</Prose>

<Prose>{""}<strong>{"Implementation exercise:"}</strong>{" add a positive routing temperature τ by replacing "}<code>{"softmax(logits)"}</code>{" with "}<code>{"softmax(logits/τ)"}</code>{", leaving the agreement update unscaled. Compare τ0.5,1 and2 with fixed votes and R. Do not divide both the logits and every agreement update unless you mean a different algorithm. The worked two-parent logits[0,2] give shares approximately[0.1192,0.8808] atτ1, [0.0180,0.9820] atτ0.5, and[0.2689,0.7311] atτ2. Check each child row sums to one and that a single candidate parent always receives share one. Trace actual vector outputs as well as shares; sharper assignments do not guarantee better classification. At finite nonzero votes this remains differentiable, while τ must remain strictly positive."}</Prose>

<H2>{"8. Architecture costs and alternative routing designs"}</H2>

<Prose>{"The classic vector CapsNet uses a larger shape chain than our experiment:"}</Prose>

<CapsuleClassicShape />

<div className="neural-equation"><MathBlock>{"28^2\\to256\\times20\\times20\n\\to32\\text{ types}\\times6\\times6\\times8\\text{ coordinates}\n\\to10\\times16."}</MathBlock></div>

<Prose>{"Both convolutions have "}<InlineMath>{"9\\times9"}</InlineMath>{" kernels; the second has stride 2. There are 1,152 primary capsules. With a masked 160-value decoder input and decoder widths 512, 1,024 and 784, the parameter accounting is:"}</Prose>

<NeuralTable caption={"8. Architecture costs and alternative routing designs"} headers={[<>{"Component"}</>,<>{"Parameters"}</>]} rows={[[<>{"First convolution"}</>,<>{"20,992"}</>],[<>{"Primary-capsule convolution"}</>,<>{"5,308,672"}</>],[<>{"Vote transformations"}</>,<>{"1,474,560"}</>],[<>{"Reconstruction decoder"}</>,<>{"1,411,344"}</>],[<>{"Total"}</>,<>{"8,215,568"}</>]]} />

<Prose>{"The primary convolution, not the vote matrices, owns the largest share here. Disabling the decoder removes its parameters. A decoder that receives only the selected 16-vector is another design with different counts; do not mix that count with the masked-160 implementation."}</Prose>

<Prose>{"For historical context, the vector-capsule paper reports .25% ordinary MNIST test error for its three-routing-step reconstruction model. Its 99.23% MNIST accuracy belongs to a different, expanded-canvas model used in the affNIST transfer comparison. Those are not interchangeable results. Matrix-capsule experiments also use smallNORB: photographs of physical toy objects under controlled views and lighting, not rendered 3D objects. Its separation of physical training and test instances is material to interpreting generalization. "}<a href={"https://arxiv.org/pdf/1710.09829"}>{"Vector-capsule experiments"}</a>{", "}<a href={"https://www.cs.toronto.edu/~hinton/absps/EMcapsules.pdf"}>{"matrix-capsule experiments"}</a>{"."}</Prose>

<Prose>{"For "}<InlineMath>{"I"}</InlineMath>{" children, "}<InlineMath>{"J"}</InlineMath>{" parents, child dimension "}<InlineMath>{"d"}</InlineMath>{", parent dimension "}<InlineMath>{"D"}</InlineMath>{" and "}<InlineMath>{"r"}</InlineMath>{" routing steps:"}</Prose>

<ul><li>{"Dense vote parameters and vote products scale with "}<InlineMath>{"IJdD"}</InlineMath>{"."}</li><li>{"Stored votes scale with "}<InlineMath>{"BIJD"}</InlineMath>{", where "}<InlineMath>{"B"}</InlineMath>{" is batch size."}</li><li>{"Weighted sums at all "}<InlineMath>{"r"}</InlineMath>{" steps plus agreements at the first "}<InlineMath>{"r-1"}</InlineMath>{" steps require "}<InlineMath>{"(2r-1)BIJD"}</InlineMath>{" scalar product-and-accumulate terms, excluding softmax, squash and other operations."}</li></ul>

<Prose>{"At batch 32, the classic vote tensor alone occupies 23,592,960 bytes in float32, about 22.5 MiB. Backpropagation retains additional state. Arithmetic counts do not predict latency without measuring memory movement, kernels, hardware and backward computation."}</Prose>

<Prose>{"Naively expanding the original valid-convolution topology to "}<InlineMath>{"224\\times224"}</InlineMath>{" gives a "}<InlineMath>{"104\\times104"}</InlineMath>{" primary grid, or 346,112 children. Fully connecting these to 1,000 parents with dimensions 8→16 would require 44,302,336,000 vote parameters. This is a warning about that particular expansion, not a lower bound for every capsule architecture. Local capsule receptive fields and shared type-to-type transforms change the scaling."}</Prose>

<Prose>{"Matrix multiplication of a "}<InlineMath>{"4\\times4"}</InlineMath>{" pose by a learned "}<InlineMath>{"4\\times4"}</InlineMath>{" relation uses 16 learned parameters per relation. An unrestricted linear map of a flattened 16-vector to another 16-vector uses 256. That parameter reduction imposes structure; it is not a free replacement for every arbitrary vector map."}</Prose>

<Prose>{"Three alternative directions answer different shortcomings:"}</Prose>

<NeuralTable caption={"8. Architecture costs and alternative routing designs"} headers={[<>{"Direction"}</>,<>{"Mechanism"}</>,<>{"What to examine"}</>]} rows={[[<>{"Diagonal EM routing"}</>,<>{"Estimate vote clusters, spread and activation separately"}</>,<>{"Variance floors, negligible mass, log-domain numerics, local sharing"}</>],[<>{"Variational-Bayes routing"}</>,<>{"Maintain approximate uncertainty over mixture parameters and assignments with priors"}</>,<>{"Prior strength, approximation assumptions, whether variance-collapse behavior improves"}</>],[<>{"STAR-Caps"}</>,<>{"Use learned attentive coefficients and binary routing gates with a straight-through gradient estimator"}</>,<>{"Discrete forward choices versus surrogate gradients, actual sparse execution and measured cost"}</>]]} />

<Prose>{"The "}<a href={"https://ojs.aaai.org/index.php/AAAI/article/view/5785"}>{"AAAI 2020 variational-routing paper"}</a>{" and "}<a href={"https://karim-ahmed.github.io/publications/starcaps.pdf"}>{"NeurIPS 2019 STAR-Caps paper"}</a>{" are distinct algorithms, not extra loop counts for the vector-routing function above. STAR-Caps includes ImageNet experiments, so “capsules have never been tried on ImageNet” is incorrect. Historical results should be read with their architecture, data and training conditions, not used as an undated ranking."}</Prose>

<Prose>{"For matrix capsules, "}<strong>{"spread loss"}</strong>{" is another objective:"}</Prose>

<div className="neural-equation"><MathBlock>{"L=\\sum_{i\\ne t}\\max(0,m-(a_t-a_i))^2."}</MathBlock></div>

<Prose>{"It asks the true activation to exceed each wrong activation by a margin "}<InlineMath>{"m"}</InlineMath>{", often increased during training. Unlike the earlier independent thresholds .9 and .1, it penalizes a relative activation gap."}</Prose>

<Prose>{"Applications involving geometric structure or overlapping instances can justify capsule experiments. They still need matched baselines, valid splits and a specific failure hypothesis. Neither an attractive reconstruction nor resistance to one attack proves general robustness. A 3D viewpoint change can reveal or hide surfaces; it is not always an invertible 2D image transform."}</Prose>

<H2>{"9. Practice: implement, explain and compare"}</H2>

<H3>{"1. Repeated evidence is not an average"}</H3>

<Prose>{"There are two identical children. Each sends "}<InlineMath>{"(1,0)"}</InlineMath>{" to both of two parents. What is each parent's length after one step? Add two more identical children. What changes, and will further routing break the symmetry?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Each child splits its own contribution in half. Sum contributions at a parent before applying squash."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"Two children give "}<InlineMath>{"s=(1,0)"}</InlineMath>{", so the length is .5. Four give "}<InlineMath>{"s=(2,0)"}</InlineMath>{", so the length is .8. Both parents are identical at every step and each row remains "}<InlineMath>{"(.5,.5)"}</InlineMath>{". More evidence changes magnitude; it does not create a reason to prefer either parent."}</Prose>

</details>

<H3>{"2. A positive agreement can still lose share"}</H3>

<Prose>{"A child currently has logits "}<InlineMath>{"(0,0)"}</InlineMath>{". The next agreements are "}<InlineMath>{"(1,2)"}</InlineMath>{". Did the first parent gain or lose coupling, even though its agreement was positive?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Softmax compares scores within the same row. Compute the first fraction from the updated logits."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"Its coupling falls from .5 to "}<InlineMath>{"e^1/(e^1+e^2)=1/(1+e)\\approx.26894"}</InlineMath>{". A positive absolute update is not necessarily a relative gain."}</Prose>

</details>

<H3>{"3. Change the loss convention correctly"}</H3>

<Prose>{"A "}<InlineMath>{"16\\times16"}</InlineMath>{" reconstruction uses .0005 times summed squared error. Your API returns mean squared error over pixels. What coefficient preserves the objective? If the true class has length .8 and two wrong classes have lengths .2 and .4, compute the margin loss."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"There are 256 pixels. The true-class shortfall and the two wrong-class excesses use different weights."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"Use "}<InlineMath>{".0005\\cdot256=.128"}</InlineMath>{" times pixel MSE. The margin terms are "}<InlineMath>{"(.9-.8)^2+.5(.2-.1)^2+.5(.4-.1)^2=.01+.005+.045=.06"}</InlineMath>{", assuming all other wrong lengths are at most .1."}</Prose>

</details>

<H3>{"4. Repair a leaking reconstruction report"}</H3>

<Prose>{"A program evaluates classification without labels, but reconstructs every development image using its known true class and labels the result “inference reconstruction.” Rewrite the protocol and specify which numbers to retain."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Separate the decision available to the deployed model from an optional diagnostic that conditions on the answer."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"Select the longest class capsule to compute ordinary inference reconstruction. Retain the true-label-masked reconstruction under an explicit “label-conditioned diagnostic” label. Report both MSEs and classification errors if the distinction is useful. Neither reconstruction should change classification scores. Do not supply the true label to select a capsule in a claimed label-free system."}</Prose>

</details>

<H3>{"5. Design a new routing comparison"}</H3>

<Prose>{"You want to know whether routing helps with left-shifted digits. The existing table contains only right and downward shifts. Specify a comparison that does not treat those table rows as a new untouched test, then make and check a prediction with the program."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"State which weights are fixed, how pixels leaving the image are handled, which labels remain valid, and what data have already influenced your choices."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"One valid development investigation holds each of the six models fixed, applies a one-pixel left shift with zero fill, inspects label-preservation failures and compares paired one-step/three-step training configurations. Show the current computed result and its contributing terms immediately. The result is exploratory because the dataset and models have already been studied. A subsequent final claim needs a separately reserved, relevant evaluation set and a frozen protocol. The numeric left-shift result is intentionally not supplied: generate it, retain the changed inputs and explain both counts and disagreement cases."}</Prose>

</details>

<H3>{"6. Disprove an equivariance claim"}</H3>

<Prose>{"An engineer says any learned linear vote map preserves rotation because it is a matrix. Use "}<InlineMath>{"u=(1,0)"}</InlineMath>{", "}<InlineMath>{"W=\\operatorname{diag}(3,1)"}</InlineMath>{" and a "}<InlineMath>{"90^\\circ"}</InlineMath>{" rotation to test the claim. What must replace that assertion?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Compare transforming before the vote map with transforming afterward."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{""}<InlineMath>{"WRu=W(0,1)=(0,1)"}</InlineMath>{", while "}<InlineMath>{"RWu=R(3,0)=(0,3)"}</InlineMath>{". The diagram does not commute. Specify input/output group actions and constrain "}<InlineMath>{"W\\rho_{\\text{in}}=\\rho_{\\text{out}}W"}</InlineMath>{"; also verify the encoder, nonlinearities, routing and readout preserve the intended transformation contract."}</Prose>

</details>

<H3>{"7. A low-activation child and diagonal EM"}</H3>

<Prose>{"For one parent, two scalar votes are 0 and 4. Responsibilities are both .5; child activations are 1 and .25. Calculate effective mass, mean and variance before a variance floor. Why is repeatedly multiplying the responsibilities by activation in place wrong?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Use effective weights .5 and .125. Variance measures squared distance from the weighted mean."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"Mass is .625, mean is "}<InlineMath>{".125\\cdot4/.625=.8"}</InlineMath>{", and variance is "}<InlineMath>{"[.5(.8)^2+.125(3.2)^2]/.625=2.56"}</InlineMath>{". Repeated in-place multiplication would turn the second activation factor into .25², .25³ and so on, changing the specified model. Each iteration uses the new normalized "}<InlineMath>{"R"}</InlineMath>{" and multiplies by the unchanged "}<InlineMath>{"a"}</InlineMath>{" once."}</Prose>

</details>

<H3>{"8. Explain a reconstruction edit without inventing semantics"}</H3>

<Prose>{"Use either saved image and the fixed seed-1 three-step model. Change a selected class-vector coordinate by a value other than the demonstrated ±.1. Predict what changes and what must remain invariant. What evidence would justify naming the coordinate “stroke thickness”?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Separate encoder scores, decoder inputs, mask selection and visual interpretation."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"A decoder-only edit leaves the original encoder's scores unchanged unless you explicitly recompute a score from the edited vector. Editing a masked-out class leaves reconstruction unchanged; editing the selected class can change it. Record the exact coordinate, delta, mask and image difference. Naming a physical factor requires consistent, controlled changes across relevant images and checks for confounded properties, not one appealing morph."}</Prose>

</details>

<Prose>{""}<strong>{"Readiness:"}</strong>{" you can follow one image through grouping, voting, routing, scoring and reconstruction; explain why routing normalizes over parents for each child; distinguish input-dependent assignments from trained weights; and design a comparison whose conclusion matches the data. For the deeper branch, derive a squash sensitivity, check a transformation identity and trace one EM update."}</Prose>

<H2>{"10. Continue and learn another way"}</H2>

<Prose>{"Next in the module is "}<a href={"/learn/path/full-curriculum/rnns-lstms-grus?module=deep-learning-fundamentals"}>{"RNNs, LSTMs and GRUs"}</a>{". We move from repeated assignment refinement for one image to a state updated over observations in time. The connection is repeated computation; the purpose and state lifetime are different."}</Prose>

<Prose>{"Useful references and alternate routes:"}</Prose>

<ul><li>{""}<a href={"https://arxiv.org/abs/1710.09829"}>{"Dynamic Routing Between Capsules — Sabour, Frosst and Hinton"}</a>{". Read the routing procedure with the child/parent axes beside it, then the reconstruction experiment. Its ordinary MNIST result and the separately trained affNIST-transfer model are different protocols; do not merge their scores."}</li><li>{""}<a href={"https://www.cs.toronto.edu/~saaraa/CapsuleSlides.pdf"}>{"Introduction to Capsules — Sara Sabour's slides"}</a>{". A visual route through coordinate frames, agreement and assignment, especially slides 10–40. Some slides use cosine terminology; the implemented vector-routing agreement in this lesson is the dot product."}</li><li>{""}<a href={"https://www.crcv.ucf.edu/cvpr2019-tutorial/"}>{"Capsule Networks for Computer Vision — UCF CVPR 2019 tutorial"}</a>{". The historical university index links talks and slides (it returned a gateway error during the September2026 implementation check; use the accessible author slides above if unavailable), including Sabour's introduction, a survey, video capsules and segmentation. It is historical research teaching, with separate prerequisites for the application sessions; the full video was not watched for this packet."}</li><li>{""}<a href={"https://www.cs.toronto.edu/~hinton/absps/EMcapsules.pdf"}>{"Matrix Capsules with EM Routing"}</a>{". Follow the pose/activation distinction and algorithm, then Appendix A for why transforming Gaussian votes differ from fitting an ordinary mixture."}</li><li>{""}<a href={"https://proceedings.mlr.press/v101/paik19a.html"}>{"Capsule Networks Need an Improved Routing Algorithm"}</a>{". An alternative reading centered on controlled comparisons and assignment polarization. Its experiments are evidence under those configurations, not a universal impossibility theorem."}</li><li>{""}<a href={"https://ojs.aaai.org/index.php/AAAI/article/view/5785"}>{"Capsule Routing via Variational Bayes"}</a>{". A deeper probabilistic route; read after the local EM bridge and prior Gaussian-mixture material."}</li><li>{""}<a href={"https://karim-ahmed.github.io/publications/starcaps.pdf"}>{"STAR-Caps"}</a>{". Study the distinction between a discrete routing decision and its straight-through training gradient before attempting to reproduce this architecture."}</li><li>{""}<a href={"/learn-assets/capsule-networks/data-provenance.md"}>{"Data and calculation provenance"}</a>{", "}<a href={"/learn-assets/capsule-networks/capsule-learning.py"}>{"complete learning program"}</a>{", "}<a href={"/learn-assets/capsule-networks/capsule-mechanics.py"}>{"constructed mechanics program"}</a>{" and its "}<a href={"/learn-assets/capsule-networks/capsule_learning_import.py"}>{"small import helper"}</a>{". The first trains the offline model; the second supplies exact routing, geometry, derivative and EM fixtures. No browser lab is required to inspect the underlying arithmetic."}</li></ul>
</div>
};
