// Complete prepared revision-3 manuscript with live mechanism investigations.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { PenRepresentationLab, RecurrentCreditLab, LstmMemoryLab, LstmRetentionLab, GruResetLab, RecurrentPenLab, RecurrentBoundaryLab, RecurrentPaddingLab, RecurrentMeasuredLab, RecurrentProgram } from '../../components/lesson-labs/RecurrentLabs.jsx';
export default {
  title: 'RNNs, LSTMs & GRUs',
  readTime: '~70 min read + experiments and practice',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson recurrent-lesson">
    <LessonIntro prerequisites="Vectors, affine transforms, activations and the chain rule. Sequence axes, gates and state ownership are introduced here." sections={[["1-keep-a-running-description","1. Keep a running description"],["2-build-and-train-the-simplest-recurrence","2. Build and train the simplest recurrence"],["3-lstm-retain-write-then-expose","3. LSTM: retain, write, then expose"],["4-gru-blend-old-state-with-a-new-proposal","4. GRU: blend old state with a new proposal"],["5-recognize-real-pen-trajectories","5. Recognize real pen trajectories"],["6-state-belongs-to-a-stream","6. State belongs to a stream"],["7-different-lengths-stacks-and-directions","7. Different lengths, stacks and directions"],["translate-gate-equations-into-a-reusable-recurrent-implementation","Translate gate equations into a reusable recurrent implementation"],["8-deeper-questions-and-practical-model-choices","8. Deeper questions and practical model choices"],["9-practice-change-the-problem-before-checking-the-answer","9. Practice: change the problem before checking the answer"],["10-continue-and-learn-another-way","10. Continue and learn another way"]]}>Trace a sequence into state, follow how it learns, and control what crosses a boundary.</LessonIntro>
<Prose>{""}<strong>{"Explore as you read."}</strong>{" Edit sequence entries, recurrent weights, LSTM gates, GRU reset placement and supported pen-trajectory coordinates. Update state trajectories, retained/injected terms, shared-weight credit and exact learned outputs. Step, rewind and reset state explicitly; padding and request boundaries remain visible. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose what must persist or reset, and diagnose saturation, reset-order differences and accidental cross-sequence leakage."}</Prose>

<Prose>{"A pen stroke is more than a collection of points. The order tells you how the pen travelled between them. A recurrent neural network processes that order by repeatedly updating a small collection of numbers: its current state. An LSTM or GRU changes the update rule so that the network can selectively retain, replace and expose information."}</Prose>

<Prose>{""}<strong>{"Your first pass:"}</strong>{" follow the pen trajectory and the three-step calculation in sections 1–2; learn the retain/write/read roles in sections 3–4; run or inspect the complete handwriting experiment in section 5; then work through state boundaries and padding in sections 6–7. Finish the core practice. Section 8 opens the deeper derivative and architecture questions when you are ready. You do not need its full Jacobian derivation to understand the next lesson."}</Prose>

<Prose>{"The previous "}<a href={"/learn/path/full-curriculum/capsule-networks?module=deep-learning-fundamentals"}>{"Capsule Networks lesson"}</a>{" organized parts inside one input. Here we follow observations across positions in a sequence. The distinction matters: routing iterations within a capsule model are not elapsed time, and the hidden state of an RNN is not a routing coefficient."}</Prose>

<H2>{"1. Keep a running description"}</H2>

<Prose>{"Imagine identifying a handwritten digit from eight successive pen coordinates. After point 1, you know where the trace starts. After point 4, you have evidence about a bend. By point 8, the model must produce one of ten digit labels."}</Prose>

<Prose>{"One approach is to concatenate all sixteen coordinates and use an ordinary classifier. That preserves their order: the first pair and last pair occupy different input columns. Another approach summarizes the points by their mean, spread and extremes; it loses the order. A recurrent model offers a third approach: apply the same update rule to each new point and a state summarizing the previous points."}</Prose>

<Prose>{"This distinction is useful beyond handwriting. A recurrent state can summarize successive machine measurements, maintain context during a conversation, or track what a controller has observed. The state is learned for the prediction objective. It is not a lossless recording of every past event, and it does not automatically represent a human-readable fact."}</Prose>

<PenRepresentationLab />

<Prose>{"Let "}<InlineMath>{"x_t"}</InlineMath>{" be the observation at position "}<InlineMath>{"t"}</InlineMath>{". A state "}<InlineMath>{"h_t"}</InlineMath>{" is a vector produced after observing that input:"}</Prose>

<div className="neural-equation"><MathBlock>{"h_t=F_\\theta(x_t,h_{t-1})."}</MathBlock></div>

<Prose>{"The same parameters "}<InlineMath>{"\\theta"}</InlineMath>{" appear at every position. During one prediction, these weights stay fixed while the state changes. Training changes the weights between optimizer updates. Mixing up these two changes makes recurrent models seem more mysterious than they are."}</Prose>

<Prose>{"For our pen example, each "}<InlineMath>{"x_t"}</InlineMath>{" has two coordinates and "}<InlineMath>{"h_t"}</InlineMath>{" will have 32 learned features. A batch of 64 complete traces has shape "}<InlineMath>{"[64,8,2]"}</InlineMath>{": specimen, position, coordinate. At one position the state has shape "}<InlineMath>{"[64,32]"}</InlineMath>{". Rows belong to different specimens and must not exchange their histories."}</Prose>

<Prose>{"There are several useful output arrangements:"}</Prose>

<NeuralTable caption={"1. Keep a running description"} headers={[<>{"Input and output"}</>,<>{"Example"}</>,<>{"Which state is read?"}</>]} rows={[[<>{"Many observations → one label"}</>,<>{"A complete pen trace → digit"}</>,<>{"A summary after the valid final point"}</>],[<>{"One prediction per observation"}</>,<>{"A sensor prefix → current operating state"}</>,<>{"Each causal state "}<InlineMath>{"h_t"}</InlineMath>{""}</>],[<>{"Previous symbols → next symbol"}</>,<>{"A token prefix → a next-token distribution"}</>,<>{"State after the preceding symbols"}</>],[<>{"One sequence → another"}</>,<>{"A source sentence → a translation"}</>,<>{"An encoder and a separate decoder"}</>]]} />

<Prose>{"The first is our complete experiment. The last is the "}<a href={"/learn/path/full-curriculum/sequence-to-sequence-encoder-decoder?module=deep-learning-fundamentals"}>{"next lesson"}</a>{". A recurrent layer is a reusable component, not a complete specification of what a model predicts."}</Prose>

<H3>{"Position does not necessarily mean seconds"}</H3>

<Prose>{"The real data used below contains eight points resampled at approximately equal distances along each completed pen trace. Their positions preserve order but are not eight equally spaced timestamps. An input collected at irregular times may need elapsed-time features as well as its measurements. A recurrence knows which observation came first; it cannot infer an unrecorded time gap."}</Prose>

<H2>{"2. Build and train the simplest recurrence"}</H2>

<Prose>{"A simple tanh RNN first mixes the new input with the old state, then applies a bounded nonlinear transformation:"}</Prose>

<div className="neural-equation"><MathBlock>{"a_t=W_xx_t+W_hh_{t-1}+b,\\qquad h_t=\\tanh(a_t)."}</MathBlock></div>

<Prose>{""}<InlineMath>{"W_x"}</InlineMath>{" maps input features into state features. "}<InlineMath>{"W_h"}</InlineMath>{" mixes the previous state features with each other. "}<InlineMath>{"b"}</InlineMath>{" shifts the resulting values. The tanh function maps any real number into "}<InlineMath>{"(-1,1)"}</InlineMath>{"; near zero it is close to its input, but large positive or negative inputs produce values close to the endpoints."}</Prose>

<Prose>{"For "}<InlineMath>{"D"}</InlineMath>{" input features and "}<InlineMath>{"H"}</InlineMath>{" state features, "}<InlineMath>{"W_x"}</InlineMath>{" is "}<InlineMath>{"H\\times D"}</InlineMath>{", "}<InlineMath>{"W_h"}</InlineMath>{" is "}<InlineMath>{"H\\times H"}</InlineMath>{", and "}<InlineMath>{"b"}</InlineMath>{" has "}<InlineMath>{"H"}</InlineMath>{" entries. A classification head converts the final state into scores "}<InlineMath>{"Wh_T+b_{\\text{out}}"}</InlineMath>{", then softmax converts scores to class probabilities. The matrices have no time index because they are shared."}</Prose>

<Prose>{"The "}<a href={"https://d2l.ai/chapter_recurrent-neural-networks/rnn.html"}>{"D2L recurrent-network explanation"}</a>{" develops this distinction between a hidden layer and a hidden state. Here we can calculate the whole chain rather than relying on the diagram."}</Prose>

<H3>{"Three observations, one state number"}</H3>

<Prose>{"Use "}<InlineMath>{"x=[0.4,-0.2,0.7]"}</InlineMath>{", input weight "}<InlineMath>{"w_x=0.8"}</InlineMath>{", recurrent weight "}<InlineMath>{"w_h=0.6"}</InlineMath>{", bias "}<InlineMath>{"b=0.1"}</InlineMath>{", and "}<InlineMath>{"h_0=0"}</InlineMath>{". These are deliberately constructed numbers, not fitted handwriting features."}</Prose>

<NeuralTable caption={"Three observations, one state number"} headers={[<>{"Position"}</>,<>{"New input"}</>,<>{"Input contribution "}<InlineMath>{"0.8x_t"}</InlineMath>{""}</>,<>{"Previous-state contribution "}<InlineMath>{"0.6h_{t-1}"}</InlineMath>{""}</>,<>{""}<InlineMath>{"h_t"}</InlineMath>{" after adding bias and tanh"}</>]} rows={[[<>{"1"}</>,<>{"0.4"}</>,<>{"0.32"}</>,<>{"0"}</>,<>{"0.396930"}</>],[<>{"2"}</>,<>{"−0.2"}</>,<>{"−0.16"}</>,<>{"0.238158"}</>,<>{"0.176297"}</>],[<>{"3"}</>,<>{"0.7"}</>,<>{"0.56"}</>,<>{"0.105778"}</>,<>{"0.644468"}</>]]} />

<Prose>{"The second input is negative, yet the second state is positive. The state includes both the new observation and a transformed contribution from the past. It is not simply a copy of the current input."}</Prose>

<RecurrentCreditLab />

<H3>{"Learning assigns credit to repeated uses of the same weight"}</H3>

<Prose>{"Suppose the desired final state is "}<InlineMath>{"y=0.3"}</InlineMath>{" and the loss is"}</Prose>

<div className="neural-equation"><MathBlock>{"L=\\frac12(h_3-y)^2."}</MathBlock></div>

<Prose>{"The calculated loss is "}<InlineMath>{"0.059329"}</InlineMath>{". To reduce it, training needs to know how changing each weight would change the final output."}</Prose>

<Prose>{"Unroll the recurrence into three copies of the calculation. There are three state values, but still one shared input weight, one shared recurrent weight and one shared bias. "}<strong>{"Backpropagation through time (BPTT)"}</strong>{" is ordinary chain-rule backpropagation through this unrolled graph."}</Prose>

<Prose>{"Start with "}<InlineMath>{"\\partial L/\\partial h_3=h_3-y=0.344468"}</InlineMath>{". Since the derivative of tanh at "}<InlineMath>{"a_t"}</InlineMath>{" is "}<InlineMath>{"1-h_t^2"}</InlineMath>{", the credit arriving at the preactivation is"}</Prose>

<div className="neural-equation"><MathBlock>{"\\delta_t=\\frac{\\partial L}{\\partial h_t}(1-h_t^2)."}</MathBlock></div>

<Prose>{"Each use contributes "}<InlineMath>{"\\delta_tx_t"}</InlineMath>{" to the input-weight gradient, "}<InlineMath>{"\\delta_th_{t-1}"}</InlineMath>{" to the recurrent-weight gradient, and "}<InlineMath>{"\\delta_t"}</InlineMath>{" to the bias gradient. The previous state receives "}<InlineMath>{"w_h\\delta_t"}</InlineMath>{". Move backward and add the contributions from every use:"}</Prose>

<NeuralTable caption={"Learning assigns credit to repeated uses of the same weight"} headers={[<>{"Shared parameter"}</>,<>{"Contribution at step 3"}</>,<>{"At step 2"}</>,<>{"At step 1"}</>,<>{"Total"}</>]} rows={[[<>{""}<InlineMath>{"w_x"}</InlineMath>{""}</>,<>{"0.140978"}</>,<>{"−0.023416"}</>,<>{"0.023673"}</>,<>{"0.141234"}</>],[<>{""}<InlineMath>{"w_h"}</InlineMath>{""}</>,<>{"0.035506"}</>,<>{"0.046474"}</>,<>{"0"}</>,<>{"0.081979"}</>],[<>{""}<InlineMath>{"b"}</InlineMath>{""}</>,<>{"0.201397"}</>,<>{"0.117082"}</>,<>{"0.059181"}</>,<>{"0.377661"}</>]]} />

<Prose>{"With learning rate "}<InlineMath>{"0.1"}</InlineMath>{", subtract "}<InlineMath>{"0.1"}</InlineMath>{" times each gradient. The updated parameters are approximately "}<InlineMath>{"(0.785877,0.591802,0.062234)"}</InlineMath>{"; recalculating the sequence gives loss "}<InlineMath>{"0.042839"}</InlineMath>{". The retained calculation agrees with automatic differentiation. This single successful step explains the mechanism; it does not establish that any learning rate will improve every step."}</Prose>

<Prose>{"If there are losses at several positions, add each position's direct loss gradient to the gradient arriving from the future before propagating backward. The "}<a href={"/learn/path/full-curriculum/backpropagation-automatic-differentiation?module=deep-learning-fundamentals"}>{"Backpropagation & Automatic Differentiation lesson"}</a>{" supplies the general graph perspective."}</Prose>

<H3>{"Why distant credit can become difficult"}</H3>

<Prose>{"For a vector state, the local derivative with respect to the previous state is"}</Prose>

<div className="neural-equation"><MathBlock>{"J_t=\\operatorname{diag}(1-h_t^2)W_h."}</MathBlock></div>

<Prose>{"Credit from a distant position involves a product of these matrices. Repeated contraction can make an earlier input's effect tiny; repeated expansion along relevant directions can make gradients very large. The actual activations and directions matter. A large recurrent matrix alone does not prove exploding gradients."}</Prose>

<Prose>{"For example, a scalar recurrent weight of 2 at preactivation 5 gives local derivative "}<InlineMath>{"2(1-\\tanh^2 5)\\approx0.000363"}</InlineMath>{", a strong contraction. The tanh saturation overwhelms the weight. Section 8 separates reliable norm bounds from misleading eigenvalue shortcuts. "}<a href={"https://proceedings.mlr.press/v28/pascanu13.pdf"}>{"Pascanu, Mikolov and Bengio"}</a>{" analyze temporal gradient products and motivate gradient-norm clipping."}</Prose>

<Prose>{"Clipping limits the size of a gradient that is already available. It cannot reconstruct a signal that has vanished. Gated recurrences instead change the pathways through which state and credit travel."}</Prose>

<H2>{"3. LSTM: retain, write, then expose"}</H2>

<Prose>{"An LSTM carries two vectors. The "}<strong>{"cell state"}</strong>{" "}<InlineMath>{"c_t"}</InlineMath>{" has an additive update path. The "}<strong>{"hidden state"}</strong>{" "}<InlineMath>{"h_t"}</InlineMath>{" is a transformed, gated view of that cell and is also used to compute the next update. Calling them “long-term” and “short-term” memory can be a starting analogy, but neither name guarantees a retention duration."}</Prose>

<Prose>{"Before looking at the learned gates, consider one coordinate:"}</Prose>

<div className="neural-equation"><MathBlock>{"c_t=f_tc_{t-1}+i_tg_t,\\qquad h_t=o_t\\tanh(c_t)."}</MathBlock></div>

<Prose>{"Here "}<InlineMath>{"f_t"}</InlineMath>{" controls retention, "}<InlineMath>{"i_t"}</InlineMath>{" controls writing, "}<InlineMath>{"g_t"}</InlineMath>{" proposes a signed value, and "}<InlineMath>{"o_t"}</InlineMath>{" controls how much of the transformed cell is exposed. Each symbol is one coordinate here; a real layer performs these operations component by component."}</Prose>

<Prose>{"If the old cell is "}<InlineMath>{"0.8"}</InlineMath>{", retention "}<InlineMath>{"f=0.9"}</InlineMath>{", input gate "}<InlineMath>{"i=0.2"}</InlineMath>{", candidate "}<InlineMath>{"g=-0.5"}</InlineMath>{", and output gate "}<InlineMath>{"o=0.6"}</InlineMath>{", then:"}</Prose>

<ol><li>{"Keep "}<InlineMath>{"0.9\\times0.8=0.72"}</InlineMath>{" from the old cell."}</li><li>{"Write "}<InlineMath>{"0.2\\times(-0.5)=-0.10"}</InlineMath>{"."}</li><li>{"Add them to get "}<InlineMath>{"c=0.62"}</InlineMath>{"."}</li><li>{"Expose "}<InlineMath>{"h=0.6\\tanh(0.62)\\approx0.330677"}</InlineMath>{"."}</li></ol>

<Prose>{"An output gate near zero can hide the cell from the current readout while leaving its stored value present. An input gate near zero can avoid writing a distracting candidate. The forget gate controls the old term; it does not itself decide what new value replaces it."}</Prose>

<LstmMemoryLab />

<H3>{"The gates are computed, not hand-selected during prediction"}</H3>

<Prose>{"The standard non-peephole LSTM used here computes four affine transforms of the current input and previous hidden state:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\begin{aligned}\ni_t&=\\sigma(W_{xi}x_t+W_{hi}h_{t-1}+b_i),\\\\\nf_t&=\\sigma(W_{xf}x_t+W_{hf}h_{t-1}+b_f),\\\\\ng_t&=\\tanh(W_{xg}x_t+W_{hg}h_{t-1}+b_g),\\\\\no_t&=\\sigma(W_{xo}x_t+W_{ho}h_{t-1}+b_o).\n\\end{aligned}"}</MathBlock></div>

<Prose>{"The sigmoid "}<InlineMath>{"\\sigma(a)=1/(1+e^{-a})"}</InlineMath>{" produces a number in "}<InlineMath>{"(0,1)"}</InlineMath>{", appropriate for a multiplicative gate. The candidate uses tanh because a proposed write may be positive or negative. There are three sigmoid gates and one candidate transform. Gate values vary across coordinates and positions, even though their learned matrices are shared. In vector equations, "}<InlineMath>{"\\odot"}</InlineMath>{" means multiply corresponding coordinates, not matrix multiplication."}</Prose>

<Prose>{"The "}<a href={"https://d2l.ai/chapter_recurrent-modern/lstm.html"}>{"D2L LSTM walkthrough"}</a>{" is a useful second account of these operations. Modern LSTM notation includes a forget gate; the "}<a href={"https://www.bioinf.jku.at/publications/older/2604.pdf"}>{"1997 LSTM paper"}</a>{" introduced an earlier architecture, and "}<a href={"https://pubmed.ncbi.nlm.nih.gov/11032042/"}>{"Gers, Schmidhuber and Cummins"}</a>{" introduced adaptive forgetting in 2000. Historical implementations and today's full automatic differentiation are not identical algorithms."}</Prose>

<H3>{"“Almost one” still compounds"}</H3>

<Prose>{"Hold the write contribution at zero and the forget factor constant. Starting at "}<InlineMath>{"c_0=1"}</InlineMath>{", the direct retained value after "}<InlineMath>{"T"}</InlineMath>{" steps is "}<InlineMath>{"f^T"}</InlineMath>{"."}</Prose>

<NeuralTable caption={"“Almost one” still compounds"} headers={[<>{"Fixed forget factor"}</>,<>{"After 10 steps"}</>,<>{"After 100 steps"}</>,<>{"Steps to halve the retained value"}</>]} rows={[[<>{"0.5"}</>,<>{"0.000977"}</>,<>{""}<InlineMath>{"7.89\\times10^{-31}"}</InlineMath>{""}</>,<>{"1"}</>],[<>{""}<InlineMath>{"\\sigma(1)\\approx0.731059"}</InlineMath>{""}</>,<>{"0.043604"}</>,<>{""}<InlineMath>{"2.48\\times10^{-14}"}</InlineMath>{""}</>,<>{"2.21"}</>],[<>{"0.99"}</>,<>{"0.904382"}</>,<>{"0.366032"}</>,<>{"68.97"}</>],[<>{"0.999"}</>,<>{"0.990045"}</>,<>{"0.904792"}</>,<>{"692.80"}</>]]} />

<Prose>{"These are exact-formula illustrations, not measured gradients of trained networks. A positive forget bias can initially favor retention, but bias 1 does not by itself preserve a signal for hundreds of steps. To retain half over 100 fixed-factor steps requires "}<InlineMath>{"f=0.5^{1/100}\\approx0.993092"}</InlineMath>{", corresponding to sigmoid preactivation about 4.968. Real gates also depend on inputs and state."}</Prose>

<LstmRetentionLab />

<Prose>{"This controllable additive route helps with learning long dependencies. It is not a promise that every LSTM gradient stays constant: gates can close, output tanh can saturate, and the complete state has additional derivative paths."}</Prose>

<H2>{"4. GRU: blend old state with a new proposal"}</H2>

<Prose>{"A GRU carries one state vector. Define its update gate "}<InlineMath>{"z_t"}</InlineMath>{" as the fraction of old state retained:"}</Prose>

<div className="neural-equation"><MathBlock>{"h_t=z_t\\odot h_{t-1}+(1-z_t)\\odot n_t."}</MathBlock></div>

<Prose>{"If an old coordinate is "}<InlineMath>{"0.8"}</InlineMath>{", its candidate is "}<InlineMath>{"-0.4"}</InlineMath>{", and "}<InlineMath>{"z=0.75"}</InlineMath>{", the new state is "}<InlineMath>{"0.75(0.8)+0.25(-0.4)=0.5"}</InlineMath>{". A large "}<InlineMath>{"z"}</InlineMath>{" means a small replacement in this convention. Some presentations use the complementary convention, so read the equation before interpreting the word “update.”"}</Prose>

<Prose>{"The reset gate "}<InlineMath>{"r_t"}</InlineMath>{" controls how the old state influences the candidate. The update gate decides how much candidate actually replaces the old state. Resetting the candidate's dependence on history is therefore not the same as clearing the entire carried state."}</Prose>

<Prose>{"For the PyTorch variant used in the program:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\begin{aligned}\nr_t&=\\sigma(W_{xr}x_t+b_{xr}+W_{hr}h_{t-1}+b_{hr}),\\\\\nz_t&=\\sigma(W_{xz}x_t+b_{xz}+W_{hz}h_{t-1}+b_{hz}),\\\\\nn_t&=\\tanh(W_{xn}x_t+b_{xn}+r_t\\odot(W_{hn}h_{t-1}+b_{hn})).\n\\end{aligned}"}</MathBlock></div>

<Prose>{"The "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html"}>{"official GRU documentation"}</a>{" specifies this reset-after-recurrent-affine form. The "}<a href={"https://arxiv.org/abs/1406.1078"}>{"original encoder–decoder paper"}</a>{" uses a reset-before-matrix form. They are not interchangeable when copying weights between implementations."}</Prose>

<Prose>{"To see the difference without an entire network, let "}<InlineMath>{"h=[1,2]^T"}</InlineMath>{", "}<InlineMath>{"r=[0.2,0.8]^T"}</InlineMath>{", "}<InlineMath>{"W=\\begin{bmatrix}1&2\\\\3&4\\end{bmatrix}"}</InlineMath>{", and recurrent bias "}<InlineMath>{"b=[0.5,-0.5]^T"}</InlineMath>{". Then"}</Prose>

<div className="neural-equation"><MathBlock>{"W(r\\odot h)+b=[3.9,6.5]^T,\\qquad\nr\\odot(Wh+b)=[1.1,8.4]^T."}</MathBlock></div>

<Prose>{"Multiplying individual coordinates before mixing them is a different operation from scaling the mixed outputs. The recurrent candidate bias is also inside the reset multiplication in the PyTorch form."}</Prose>

<GruResetLab />

<Prose>{"Matched at the same hidden width, a GRU has three affine groups and a standard LSTM has four. That gives a smaller recurrent parameter count under the same bias convention. It does not establish a universal runtime or accuracy ranking, and a GRU is not literally an LSTM with one gate deleted."}</Prose>

<H2>{"5. Recognize real pen trajectories"}</H2>

<Prose>{"We now use actual handwriting coordinates instead of constructing a memory task whose answer is built into the setup."}</Prose>

<Prose>{"The "}<a href={"https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits"}>{"UCI Pen-Based Recognition of Handwritten Digits dataset"}</a>{" was contributed by E. Alpaydin and F. Alimoglu. Each row contains eight ordered "}<InlineMath>{"(x,y)"}</InlineMath>{" pairs and a digit label. The original collection separates writers between its training and test files. Our retained extract takes the first 60 specimens per class from the original training file and the first 30 per class from the original test file: 600 training and 300 "}<strong>{"development"}</strong>{" specimens."}</Prose>

<Prose>{"We call the second set development because we inspect multiple architectures, seeds and input changes on it. It is no longer an untouched final test. Writer identifiers are absent from individual rows, so the inherited writer separation relies on the source's documented collection procedure. Exact coordinate-vector checks found no duplicates in either full source file and no duplicates across them."}</Prose>

<Prose>{"The coordinates were normalized and resampled using each completed trace. We scale their provided range "}<InlineMath>{"0\\ldots100"}</InlineMath>{" to "}<InlineMath>{"[-1,1]"}</InlineMath>{" with "}<InlineMath>{"x/50-1"}</InlineMath>{". This is fixed arithmetic, with no learned normalization statistics. Because the source preprocessing uses the completed specimen, these results evaluate "}<strong>{"completed-trace classification"}</strong>{", not a live system observing an unfinished pen stroke."}</Prose>

<Prose>{"The "}<a href={"/learn-assets/rnns-lstms-grus/pen-trajectories.csv"}>{"offline CSV"}</a>{", "}<a href={"/learn-assets/rnns-lstms-grus/data-provenance.md"}>{"provenance"}</a>{" and "}<a href={"/learn-assets/rnns-lstms-grus/data-extraction.json"}>{"exact extraction record"}</a>{" preserve the source rows, attribution and hashes. No pretrained model or new data download is needed to run the lesson experiment."}</Prose>

<H3>{"Fix the comparison before fitting"}</H3>

<Prose>{"All three recurrent models use one unidirectional layer, input width 2, hidden width 32, and a linear ten-class head reading the final hidden output. They start at zero state for every specimen. Train each for 500 Adam updates at learning rate 0.005, batch size 64 sampled with replacement, and cross-entropy on the final digit. Clip the global gradient norm at 1 before each update. There is no dropout, augmentation, early stopping or best-seed selection."}</Prose>

<Prose>{"Seeds 1, 2 and 3 each determine a new initialization. For a given seed, the batch generator supplies the same specimen indices across the three architectures. Their different parameter shapes prevent claiming identical initial weights. Native recurrent biases are set to zero, except for an effective LSTM forget bias of 1."}</Prose>

<Prose>{"There are also two fixed logistic-regression baselines, each with scaling fitted only on training rows: one sees orderless coordinate statistics; the other sees all sixteen ordered coordinates. Neither is recurrent. They make the contribution of information representation visible."}</Prose>

<H3>{"Complete program"}</H3>

<Prose>{"Create a Python environment with NumPy, PyTorch and scikit-learn, then save the CSV beside the following file. The author run used Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.0+cpu and scikit-learn 1.9.1. Run:"}</Prose>

<CodeBlock language={"text"}>{"python pen-sequence-learning.py"}</CodeBlock>

<Prose>{"The program records every declared run, final probabilities, reversal and point-swap results, and the three seed-1 model weights in "}<code>{"calculated-inputs.json"}</code>{". It does not choose a winning model using development results."}</Prose>

<RecurrentProgram file="pen-sequence-learning.py" title="Read the complete real-data training and evaluation program" />

<Prose>{"The forward pass returns a sequence of hidden outputs and a final state. "}<code>{"outputs[:, -1]"}</code>{" is correct here because every specimen has exactly eight valid points. Section 7 explains why this expression becomes wrong for a padded batch. Cross-entropy receives raw scores; softmax is used when reporting probabilities."}</Prose>

<H3>{"What actually happened"}</H3>

<NeuralTable caption={"What actually happened"} headers={[<>{"Model"}</>,<>{"Trainable parameters"}</>,<>{"Development correct, seed 1"}</>,<>{"Seed 2"}</>,<>{"Seed 3"}</>]} rows={[[<>{"Orderless statistics + logistic regression"}</>,<>{"Baseline feature model"}</>,<>{"152/300"}</>,<>{"Same fixed baseline"}</>,<>{"Same fixed baseline"}</>],[<>{"Ordered coordinates + logistic regression"}</>,<>{"Baseline feature model"}</>,<>{"262/300"}</>,<>{"Same fixed baseline"}</>,<>{"Same fixed baseline"}</>],[<>{"Tanh RNN, 32 state features"}</>,<>{"1,482"}</>,<>{"268/300"}</>,<>{"267/300"}</>,<>{"274/300"}</>],[<>{"LSTM, 32 cell/hidden features"}</>,<>{"4,938"}</>,<>{"273/300"}</>,<>{"276/300"}</>,<>{"271/300"}</>],[<>{"GRU, 32 state features"}</>,<>{"3,786"}</>,<>{"279/300"}</>,<>{"276/300"}</>,<>{"275/300"}</>]]} />

<Prose>{"The ordered baseline already gets many examples right. Recurrence is useful to investigate, but it is not necessary merely to make order available. These short traces also do not test whether a model can retain a cue over hundreds of steps."}</Prose>

<Prose>{"Seed-1 development cross-entropies are approximately 0.3904 for RNN, 0.2629 for LSTM and 0.2611 for GRU; lower is better. The corresponding training correct counts are 597, 595 and 594 out of 600. Accuracy and cross-entropy measure different aspects: changing a probability can change the loss without changing the most likely digit."}</Prose>

<Prose>{"All nine runs remain visible in the retained results. A few additional correct specimens in this small, inspected development set do not settle which architecture will work best on another task. No latency was measured, so these numbers support no hardware-speed claim."}</Prose>

<RecurrentMeasuredLab />

<H3>{"Reverse the trace while keeping the fitted weights"}</H3>

<Prose>{"The orderless model returns the same probabilities to floating-point precision when the eight points are reversed. The ordered baseline falls from 262 to 22 correct out of 300. The seed-1 RNN, LSTM and GRU fall from 268, 273 and 279 to 42, 39 and 37 respectively."}</Prose>

<Prose>{"The digit label is retained for this constructed input transformation. The reversal uses the same set of points but changes traversal direction, including start and end. It is an explicit distribution-change probe, not a fresh natural handwriting benchmark. Reversing inputs at evaluation is also different from retraining on reversed inputs."}</Prose>

<Prose>{"Swapping only points 3 and 4 gives a milder but still meaningful change: seed-1 correct counts become 262, 239 and 259. The models depend on more than the unconnected point set."}</Prose>

<RecurrentPenLab />

<Prose>{"The checked edit that adds 0.2 in normalized units to point 3's x-coordinate changes probabilities for the first two retained specimens but does not need to change the predicted digit. That is a useful outcome: a visible input change is not evidence that the classifier must flip its decision."}</Prose>

<Prose>{"The hidden-state readout after each intermediate point is a view into the same fitted classifier. It was trained for the final point and receives coordinates preprocessed from the complete trace. Treat those intermediate readouts as a mechanism illustration, not validated early-recognition probabilities."}</Prose>

<H2>{"6. State belongs to a stream"}</H2>

<Prose>{"If five observations belong to one continuous sequence, processing the first two and carrying their final state into the next three should reproduce processing all five together, assuming the same weights and deterministic evaluation behavior. Splitting a file into chunks does not create a new phenomenon in the data."}</Prose>

<Prose>{"Resetting at the boundary is different: it removes the preceding context. Detaching at the boundary is different again: it preserves the numerical state but prevents a later loss from sending gradients through the earlier chunk."}</Prose>

<NeuralTable caption={"6. State belongs to a stream"} headers={[<>{"Boundary operation"}</>,<>{"Forward state value"}</>,<>{"Credit to the earlier computation"}</>]} rows={[[<>{"Carry state"}</>,<>{"Preserved"}</>,<>{"Preserved if graph retained"}</>],[<>{"Carry detached state"}</>,<>{"Preserved"}</>,<>{"Cut at the boundary"}</>],[<>{"Reset state to zero"}</>,<>{"Replaced"}</>,<>{"Earlier state no longer used"}</>]]} />

<Prose>{"For a constructed five-observation GRU fixture, whole-sequence and carried-chunk outputs agree exactly in the author calculation. Resetting changes the last hidden output by up to 0.046305. Detaching preserves the forward outputs exactly, but the later loss's gradients with respect to the first two inputs become zero."}</Prose>

<Prose>{"This is "}<strong>{"truncated BPTT"}</strong>{" when applied during training across chunk boundaries. It bounds how far the current loss directly backpropagates. It does not reset memory or prove that the model cannot learn any dependency longer than a chunk. The carried states and shared parameters can still support indirect learning; the truncated gradient differs from the full-sequence objective's gradient."}</Prose>

<RecurrentBoundaryLab />

<Prose>{"For real independent pen specimens, reset for every specimen. Carrying state from one random specimen into another would make its prediction depend on an unrelated writer's previous digit. For a service managing several ongoing sessions, store state by session identity and gather the corresponding states when forming a batch. Reordering batch rows requires reordering states with their owners."}</Prose>

<Prose>{"After correcting an earlier observation, recompute the affected suffix from a valid prior state. A state cached after the old observation no longer represents the corrected prefix. A model-weight update can also invalidate a stored state if exact agreement with a fresh run under the new model is required. Practical training often carries states across parameter updates as an approximation; it should not be described as the exact full-sequence computation."}</Prose>

<H3>{"Global gradient clipping"}</H3>

<Prose>{"Treat all parameter gradients as one long vector "}<InlineMath>{"g"}</InlineMath>{". For threshold "}<InlineMath>{"\\tau>0"}</InlineMath>{","}</Prose>

<div className="neural-equation"><MathBlock>{"g_{\\text{clipped}}=g\\min\\left(1,\\frac{\\tau}{\\|g\\|_2}\\right)"}</MathBlock></div>

<Prose>{"for a nonzero finite gradient; leave the zero vector unchanged. The vector "}<InlineMath>{"[3,4]"}</InlineMath>{" has norm 5. At threshold 2, global clipping gives "}<InlineMath>{"[1.2,1.6]"}</InlineMath>{", preserving its direction. Clipping coordinates individually to "}<InlineMath>{"[-2,2]"}</InlineMath>{" gives "}<InlineMath>{"[2,2]"}</InlineMath>{", a different direction."}</Prose>

<Prose>{"Log the norm before clipping so that persistent oversized updates are visible. Clipping does not fix nonfinite gradients or recover missing long-range credit. In the handwriting runs, clipping was activated 136, 245 and 10 times for seed-1 RNN, LSTM and GRU; those counts describe this optimization setup, not an intrinsic ranking of gate quality."}</Prose>

<Prose>{"The complete "}<a href={"/learn-assets/rnns-lstms-grus/recurrent-mechanics.py"}>{"recurrent-mechanics program"}</a>{" executes the chunk, detach, padding, gate and independent native-parity calculations without refitting the handwriting models."}</Prose>

<H2>{"7. Different lengths, stacks and directions"}</H2>

<Prose>{"Most real sequence collections do not give every example the same length. Padding is an arrangement for batching; a padded zero is not automatically “no observation.” Even a zero input can change a recurrent state through its recurrent matrix and bias."}</Prose>

<Prose>{"For a right-padded unidirectional recurrence, valid outputs before the padding are unaffected by later padding. But reading the final padded position gives the state after extra updates. A backward recurrence encounters right padding before it reaches valid observations, so even its valid outputs can be contaminated."}</Prose>

<Prose>{"Packing tells the native recurrent module each sequence's actual length. Here is a complete independent API example. The short sequences are constructed to teach batching, not a new dataset experiment:"}</Prose>

<RecurrentPaddingLab />

<CodeBlock language={"python"}>{"import torch\nfrom torch import nn\nfrom torch.nn.utils.rnn import pack_padded_sequence\n\ntorch.manual_seed(8)\nsequences = [\n    torch.tensor([[.2, -.1], [.8, .3], [-.4, .9], [.7, -.2], [.1, .5]]),\n    torch.tensor([[.2, -.1], [.8, .3], [-.4, .9]]),\n    torch.tensor([[.2, -.1]]),\n]\nlengths = torch.tensor([len(sequence) for sequence in sequences])\npadded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)\nmodel = nn.GRU(2, 3, batch_first=True, bidirectional=True)\nmodel.eval()\nwith torch.no_grad():\n    packed = pack_padded_sequence(\n        padded, lengths.cpu(), batch_first=True, enforce_sorted=False)\n    _, final = model(packed)\n    # One layer: direction, specimen, feature.\n    summary = torch.cat([final[0], final[1]], dim=1)\n    separate = torch.stack([model(sequence[None])[1][:, 0]\n                            for sequence in sequences], dim=1)\nprint(padded.shape)   # torch.Size([3, 5, 2])\nprint(final.shape)    # torch.Size([2, 3, 3])\nprint(summary.shape)  # torch.Size([3, 6])\nprint(torch.allclose(final, separate, atol=1e-6))  # True"}</CodeBlock>

<Prose>{"The summary concatenates the final forward and final backward states. It does not take both halves of the output at the last valid index: at that index, the backward component has only just started reading from the end."}</Prose>

<Prose>{"The "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html"}>{"PyTorch LSTM API"}</a>{" documents the hidden/cell shapes, projections and bidirectional final-state distinction. With "}<code>{"batch_first=True"}</code>{", inputs and sequence outputs use batch first, but final hidden states still use "}<InlineMath>{"[\\text{layers}\\times\\text{directions},B,H]"}</InlineMath>{". An LSTM additionally returns a cell-state tensor. With a projection, hidden width and cell width can differ."}</Prose>

<Prose>{"In a stack, layer 2 consumes layer 1's output at each position. This adds depth across layers as well as the recurrence across positions. In a bidirectional stack, that input has both directions' features. Do not simply multiply a one-layer parameter count by the number of layers without checking the next layer's input width."}</Prose>

<Prose>{"Bidirectional models may use future observations within a completed input. That can be useful for offline labeling or an encoder that receives a whole source sequence. It is incompatible with claiming a causal output at position "}<InlineMath>{"t"}</InlineMath>{" before later observations exist. Separate the deployment question from a library flag."}</Prose>

<Prose>{"If there is a loss at every valid position, mask the padded targets too. For lengths 5, 3 and 1 there are nine valid targets, not fifteen. Dividing by fifteen dilutes the loss and changes its scale as padding changes. Packing inputs and masking output losses solve related but distinct problems."}</Prose>

<H2>{"Translate gate equations into a reusable recurrent implementation"}</H2>

<Prose>{"The complete "}<a href={"/learn-assets/rnns-lstms-grus/recurrent-mechanics.py"}>{"recurrent-mechanics.py"}</a>{" is the scratch cell owner: "}<code>{"manual_sequence"}</code>{" evaluates all RNN, LSTM and GRU gate equations in NumPy using the actual native parameter tensors. The complete "}<a href={"/learn-assets/rnns-lstms-grus/pen-sequence-learning.py"}>{"pen-sequence-learning.py"}</a>{" is the ordinary "}<code>{"nn.RNN"}</code>{", "}<code>{"nn.LSTM"}</code>{" and "}<code>{"nn.GRU"}</code>{" model route, with data preparation, loss, optimizer and evaluation. "}<code>{"scalar_credit"}</code>{" opens temporal parameter sharing and its accumulated derivative; the implemented "}<a href={"/learn/path/full-curriculum/backpropagation-automatic-differentiation"}>{"Backpropagation owner"}</a>{" owns general reverse-mode machinery."}</Prose>

<RecurrentProgram start="def manual_sequence" end="def scalar_credit" title="Read the explicit NumPy RNN, LSTM and GRU cell equations" />

<Prose>{"Read the gate ordering before copying state: LSTM uses input, forget, candidate and output slices; PyTorch GRU uses reset, update and candidate slices. There are separate input and hidden biases. In the native GRU candidate, reset multiplies the hidden affine result including its hidden bias. The earlier equation variant that resets the previous state before multiplication is not algebraically interchangeable. The supplied manual trace follows the native convention exactly and preserves both h and c for LSTM. A same-seed training score is not a gate-parity check."}</Prose>

<RecurrentProgram start="def scalar_credit" end="def state_and_padding" title="Read the scratch temporal gradient and matched autograd check" />

<Prose>{"For T steps, batch B, input I and hidden H, dense recurrence costs O(TBH(I+H)) up to the gate count. A streaming forward pass retains O(BH) hidden state (twice that for LSTM), whereas full backpropagation retains a history proportional to T. "}<code>{"manual_sequence"}</code>{" deliberately records histories for explanation; "}<code>{"nn.*"}</code>{" provides the usual batched kernels and packed-sequence route. Returned states belong to specific sequences, so reordering a batch without reordering its states is a semantic error."}</Prose>

<RecurrentProgram title="Read all native parity, chunk, padding and derivative checks" />

<Prose>{""}<strong>{"Changed-code task:"}</strong>{" process a saved trajectory in chunks of5 instead of one whole tensor. Carry the native hidden state between chunks, collect every output and concatenate along time. Compare that with one full pass using the same weights, evaluation mode and no stochastic inter-layer dropout. For LSTM carry the pair "}<code>{"(h,c)"}</code>{". Then insert "}<code>{"detach()"}</code>{" at chunk boundaries during training and explain which equality should remain and which derivative comparison should stop holding."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Detaching preserves a value but cuts its earlier computation graph; replacing a state with zeros changes the value too."}</Prose>

</details>

<details>

<summary>Solution and success criteria</summary>

<Prose>{"Chunked forward outputs agree up to numerical kernel tolerance when the state is carried correctly, including a final short chunk. Detached-state forward outputs still agree, but a final loss cannot assign credit through the detached earlier chunks. Resetting at every boundary generally changes later outputs. Test a boundary at an interior time, not only an empty or whole-length chunk. This supplies a usable streaming implementation pattern without reimplementing the cell a second time."}</Prose>

</details>

<H2>{"8. Deeper questions and practical model choices"}</H2>

<H3>{"How much does the model cost?"}</H3>

<Prose>{"For one layer, one direction, input width "}<InlineMath>{"D"}</InlineMath>{", hidden width "}<InlineMath>{"H"}</InlineMath>{", and two native bias vectors per affine group:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\begin{aligned}\n\\text{RNN parameters}&=H(D+H+2),\\\\\n\\text{GRU parameters}&=3H(D+H+2),\\\\\n\\text{LSTM parameters}&=4H(D+H+2).\n\\end{aligned}"}</MathBlock></div>

<Prose>{"Add "}<InlineMath>{"K(H+1)"}</InlineMath>{" for a "}<InlineMath>{"K"}</InlineMath>{"-class linear head. With "}<InlineMath>{"D=2,H=32,K=10"}</InlineMath>{", the recurrent counts are 1,152, 3,456 and 4,608, and the head adds 330. A textbook formula with one combined bias uses "}<InlineMath>{"+1"}</InlineMath>{" instead of "}<InlineMath>{"+2"}</InlineMath>{"; that convention is not a contradiction."}</Prose>

<Prose>{"The ordinary dense recurrent work scales with "}<InlineMath>{"T"}</InlineMath>{" times the per-step input/state matrix work. For a fixed-width unidirectional model, its carried inference state need not grow with prefix length. Full BPTT, however, retains intermediate computations for differentiation and its activation storage normally grows with sequence length. A native fused kernel can be much faster than a Python loop, but the actual runtime depends on shapes, backend and hardware."}</Prose>

<Prose>{"The standard tanh/LSTM/GRU hidden-state dependence imposes a sequential forward chain. Input projections and batch members can be processed in parallel. Later "}<a href={"/learn/path/full-curriculum/state-space-models-s4-mamba-mamba-2?module=deep-learning-fundamentals"}>{"State Space Models"}</a>{" use more structured state updates that admit other evaluation strategies. Their parallelism should not be attributed to every nonlinear recurrence."}</Prose>

<H3>{"The full LSTM derivative includes both state vectors"}</H3>

<Prose>{"If gate values and "}<InlineMath>{"h_{t-1}"}</InlineMath>{" are held fixed in a non-peephole cell, the derivative of "}<InlineMath>{"c_t"}</InlineMath>{" with respect to "}<InlineMath>{"c_{t-1}"}</InlineMath>{" is "}<InlineMath>{"\\operatorname{diag}(f_t)"}</InlineMath>{". That is the useful direct cell path."}</Prose>

<Prose>{"For the complete recurrence, the state is "}<InlineMath>{"(h,c)"}</InlineMath>{". A change in earlier cell state can also affect earlier hidden output and hence subsequent gates. The full local derivative is a block matrix:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{\\partial(h_t,c_t)}{\\partial(h_{t-1},c_{t-1})}\n=\n\\begin{bmatrix}\n\\partial h_t/\\partial h_{t-1}&\\partial h_t/\\partial c_{t-1}\\\\\n\\partial c_t/\\partial h_{t-1}&\\partial c_t/\\partial c_{t-1}\n\\end{bmatrix}."}</MathBlock></div>

<Prose>{"For the explicitly defined scalar cell in the retained mechanics program, at "}<InlineMath>{"(h,c)=(0.2,0.8)"}</InlineMath>{" it is approximately"}</Prose>

<div className="neural-equation"><MathBlock>{"\\begin{bmatrix}0.169441&0.318697\\\\0.458119&0.746494\\end{bmatrix}."}</MathBlock></div>

<Prose>{"The forget value 0.746494 appears in the lower-right entry. It is not the whole matrix. Products of these complete Jacobians describe total state sensitivities. A plot of "}<InlineMath>{"f^T"}</InlineMath>{" must therefore be labeled a direct-path, fixed-gate illustration rather than a measured full LSTM gradient."}</Prose>

<H3>{"A matrix's eigenvalues are not the whole temporal story"}</H3>

<Prose>{"For tanh, "}<InlineMath>{"\\|J_t\\|_2\\leq\\|W_h\\|_2"}</InlineMath>{". If every local Jacobian norm is bounded by a common "}<InlineMath>{"q<1"}</InlineMath>{", the product norm is at most "}<InlineMath>{"q^{T-k}"}</InlineMath>{". This is a sufficient contraction condition, not a necessary diagnosis for every trajectory."}</Prose>

<Prose>{"Time-varying matrices can behave very differently from repeating one fixed matrix. Consider"}</Prose>

<div className="neural-equation"><MathBlock>{"A=\\begin{bmatrix}0&2\\\\0&0\\end{bmatrix},\\quad\nB=\\begin{bmatrix}0&0\\\\2&0\\end{bmatrix}."}</MathBlock></div>

<Prose>{"Each has only zero eigenvalues, but "}<InlineMath>{"BA=\\operatorname{diag}(0,4)"}</InlineMath>{" amplifies one direction. This constructed counterexample explains why inspecting each step's spectral radius is insufficient for a temporal product. It is not a claim that these are the Jacobians of our fitted pen model."}</Prose>

<H3>{"Initialization, regularization and variants"}</H3>

<Prose>{"Orthogonal recurrent initialization can preserve norms for a linear transformation at initialization, but tanh derivatives and training updates still matter. The "}<a href={"/learn/path/full-curriculum/weight-initialization-xavier-kaiming-p?module=deep-learning-fundamentals"}>{"initialization lesson"}</a>{" explains the broader variance and singular-value picture."}</Prose>

<Prose>{"When setting a PyTorch LSTM forget bias, its two bias vectors add. Setting both forget slices to 1 creates effective bias 2. Our program sets the input-side slice to 1 and the recurrent-side slice to zero. The packed LSTM affine order is input, forget, candidate, output; GRU uses reset, update, candidate."}</Prose>

<Prose>{"Native recurrent-module dropout is applied between stacked layers, excluding the last layer. It is not an automatic recurrent-state dropout mechanism, and with one layer there is no intervening layer on which to apply it. A separate input/output mask or a specialized recurrent dropout method needs its own declared behavior. "}<code>{"eval()"}</code>{" changes training-dependent module behavior; "}<code>{"no_grad()"}</code>{" independently disables autograd recording."}</Prose>

<Prose>{"Peephole LSTMs let gates inspect cell state; projected LSTMs use a narrower hidden output than their cell width. These are concrete architectural choices, not synonyms for every LSTM. More specialized descendants appear later in the module."}</Prose>

<Prose>{"Choose candidate architectures from the evidence and deployment constraints. Fixed-length ordered features can be a strong baseline; a causal recurrent state can serve streaming inputs; temporal convolutions supply local receptive fields; attention supplies direct content-dependent access to other positions. No length threshold makes one architecture automatically correct. Measure quality, memory and latency under the actual input-availability contract."}</Prose>

<H2>{"9. Practice: change the problem before checking the answer"}</H2>

<H3>{"1. Alter the middle observation"}</H3>

<Prose>{"In the scalar RNN from section 2, replace "}<InlineMath>{"x_2=-0.2"}</InlineMath>{" with "}<InlineMath>{"0.2"}</InlineMath>{", leaving all other inputs and weights fixed. Which state values change? Calculate the final value."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Reuse the unchanged first state. Recompute the second preactivation, then use the new second state at step 3."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"Only "}<InlineMath>{"h_2"}</InlineMath>{" and "}<InlineMath>{"h_3"}</InlineMath>{" change. "}<InlineMath>{"h_2=\\tanh(0.16+0.6(0.396930432)+0.1)"}</InlineMath>{", and "}<InlineMath>{"h_3=\\tanh(0.56+0.6h_2+0.1)"}</InlineMath>{". The final state rises to approximately 0.733564. The earlier state cannot depend on a later input in this causal recurrence."}</Prose>

</details>

<H3>{"2. Retain a negative memory"}</H3>

<Prose>{"Let "}<InlineMath>{"c_{\\text{old}}=-0.6,f=0.8,i=0.25,g=0.4,o=0.5"}</InlineMath>{". Calculate the retained term, write term, new cell and hidden output. Would changing only "}<InlineMath>{"o"}</InlineMath>{" change the new cell?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"The output gate is applied after the cell update. Keep the signs of the two contributions separate."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"The retained term is −0.48 and the write is +0.10, so "}<InlineMath>{"c=-0.38"}</InlineMath>{" and "}<InlineMath>{"h=0.5\\tanh(-0.38)\\approx-0.18135"}</InlineMath>{". Changing only the output gate leaves this step's cell value unchanged. It changes "}<InlineMath>{"h"}</InlineMath>{", which can affect gates at later steps."}</Prose>

</details>

<H3>{"3. A reset gate is not a reset command"}</H3>

<Prose>{"A GRU coordinate has old state 0.8 and update/retain gate 0.9. After closing the candidate's reset gate, its candidate is −0.2. Is its new state zero? What update gate would retain none of the old state?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"The candidate path and the final blend are separate. Apply the blend after finding the candidate."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"The new state is "}<InlineMath>{"0.9(0.8)+0.1(-0.2)=0.7"}</InlineMath>{". Closing the reset gate does not erase the retained term. In our convention "}<InlineMath>{"z=0"}</InlineMath>{" makes the state equal to the candidate, which is still not necessarily zero."}</Prose>

</details>

<H3>{"4. Longer retention without a promise"}</H3>

<Prose>{"You want a fixed direct cell path to retain 80% after 50 steps with no writes. Derive the forget factor. Explain why setting that bias does not guarantee 80% total gradient retention in a trained LSTM."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Solve "}<InlineMath>{"f^{50}=0.8"}</InlineMath>{". Then distinguish the controlled fixed-factor experiment from input-dependent gates and the full "}<InlineMath>{"(h,c)"}</InlineMath>{" state."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{""}<InlineMath>{"f=0.8^{1/50}\\approx0.99555"}</InlineMath>{". A sigmoid preactivation "}<InlineMath>{"\\log(f/(1-f))"}</InlineMath>{" produces that factor in isolation. Real preactivations also contain input and hidden-state terms, and full gradients include other paths. The calculation describes a specified direct-path experiment."}</Prose>

</details>

<H3>{"5. The batch reordered itself"}</H3>

<Prose>{"Two ongoing sessions A and B occupy batch rows 0 and 1. The scheduler next returns rows B, A. An implementation passes its old state tensor unchanged. What is wrong, and what additional event requires more than swapping state rows?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Associate every state with the prefix that produced it, not with a permanent batch index."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"B receives A's history and A receives B's. Gather states by session identity in the new row order. If a previous observation was corrected, swapping rows is insufficient: recompute the affected state suffix from a valid prefix. Ending a session requires retiring its state before that identifier or slot is reused."}</Prose>

</details>

<H3>{"6. Padding changes the wrong result"}</H3>

<Prose>{"You batch sequences of lengths 4 and 2 with right padding. You only change the second sequence's padded values. Predict the effect on its valid forward outputs, its padded final hidden state, and its valid backward outputs."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Follow which values each direction visits before reaching the valid position."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"The valid forward outputs are unchanged. The padded final state can change because it includes the extra updates. Valid backward outputs can change because the reverse recurrence visits padding first. Packing with the true lengths removes those padding updates; reading the correct directional final states then represents the actual sequence."}</Prose>

</details>

<H3>{"7. Detach or reset?"}</H3>

<Prose>{"A later-chunk loss should use preceding context, but you can retain an autograd graph for only the current chunk. Choose carry, detach-and-carry, or reset. What agreement can you expect with a whole-sequence evaluation before any optimizer update?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Ask separately whether the numbers and the derivative connections must survive."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"Detach and carry. With identical weights and deterministic behavior, forward outputs match the unbroken recurrence. Gradients through the detached boundary do not match full BPTT. Resetting would change the forward computation too."}</Prose>

</details>

<H3>{"8. Design an honest follow-up"}</H3>

<Prose>{"You want to claim that a model recognizes a digit before the pen finishes and that it generalizes to new writers. Can you use the current intermediate probabilities as your evidence? Design the missing data and evaluation conditions."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Inspect both the time at which preprocessing can be computed and the unit kept separate by the split."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"The current complete-trace normalization/resampling uses information unavailable at a live prefix, and the classifier was trained for final-trace output. Collect or retain raw timestamped prefixes, specify causal preprocessing and the prediction time, train the declared prefix objective, keep writers separate, and reserve an untouched writer holdout after development choices. Report accuracy and uncertainty by prefix availability, with an appropriate baseline. The current results motivate this study but do not perform it."}</Prose>

</details>

<Prose>{"You are ready for the next topic when you can distinguish weights from state, execute a gated update, explain a shared-weight gradient, and keep state ownership, sequence lengths and available inputs consistent with the task. The optional full-Jacobian analysis can be revisited as you study more specialized recurrent models."}</Prose>

<H2>{"10. Continue and learn another way"}</H2>

<Prose>{"The next "}<a href={"/learn/path/full-curriculum/sequence-to-sequence-encoder-decoder?module=deep-learning-fundamentals"}>{"Sequence-to-Sequence Encoder–Decoder lesson"}</a>{" changes the output from one digit into a sequence. It introduces decoder inputs, teacher forcing, stopping and generated-prefix evaluation. The following "}<a href={"/learn/path/full-curriculum/attention-mechanism-bahdanau-luong?module=deep-learning-fundamentals"}>{"Bahdanau & Luong Attention lesson"}</a>{" lets a decoder consult source positions instead of relying only on one final summary."}</Prose>

<Prose>{"For a different explanation or a deeper reference:"}</Prose>

<ul><li>{""}<a href={"https://colah.github.io/posts/2015-08-Understanding-LSTMs/"}>{"Christopher Olah: Understanding LSTM Networks"}</a>{" is an approachable diagram-led walkthrough of cell, gate and output paths. Use it after section 3; read its strong long-memory intuition together with our fixed-gate and full-derivative qualifications."}</li><li>{""}<a href={"https://www.youtube.com/watch?v=6niqTuYFZLQ"}>{"Stanford CS231n Lecture 10: Recurrent Neural Networks"}</a>{", with "}<a href={"https://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture10.pdf"}>{"official slides"}</a>{", offers a spoken route through recurrence, language modeling, image captioning and gated models. It is a 2017 conceptual lecture, not a current framework installation guide; its captioning/attention branches lead beyond this page."}</li><li>{""}<a href={"https://d2l.ai/chapter_recurrent-neural-networks/bptt.html"}>{"D2L: Backpropagation Through Time"}</a>{" develops the gradient chain and truncation more formally. Its surrounding chapters also provide text-model implementations; our real pen example offers a different applied route."}</li><li>{""}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html"}>{"PyTorch GRU"}</a>{" and "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html"}>{"LSTM"}</a>{" document the exact native gate conventions and tensor shapes used here. Check them when transferring weights or changing directions, layers or projections."}</li><li>{""}<a href={"https://proceedings.mlr.press/v28/pascanu13.pdf"}>{"Pascanu et al., On the Difficulty of Training Recurrent Neural Networks"}</a>{" is the mathematical route to temporal gradient products and clipping. "}<a href={"https://www.bioinf.jku.at/publications/older/2604.pdf"}>{"Hochreiter & Schmidhuber's LSTM paper"}</a>{" and "}<a href={"https://arxiv.org/abs/1406.1078"}>{"Cho et al.'s encoder–decoder paper"}</a>{" provide historical mechanisms; their original algorithms and experimental claims should be read in their own settings."}</li></ul>

<Prose>{"The "}<a href={"/learn-assets/rnns-lstms-grus/calculated-inputs.json"}>{"saved numerical results"}</a>{" and "}<a href={"/learn-assets/rnns-lstms-grus/mechanics-results.json"}>{"mechanics calculations"}</a>{" separate fitted-model evidence from exact constructed examples. They let you inspect the numbers behind the lesson rather than treating an attractive plot as evidence by itself."}</Prose>
  </div>,
};
