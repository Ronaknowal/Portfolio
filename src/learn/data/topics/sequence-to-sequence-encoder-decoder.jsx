// Full revision-3 manuscript preserved; long canonical source loads on demand.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { Seq2SeqTimelines, Seq2SeqAlignmentLab, Seq2SeqBridgeLab, Seq2SeqTreeLab, Seq2SeqEvidenceFigure, Seq2SeqProgram, Seq2SeqFittedLab, seq2seqAsset } from '../../components/lesson-labs/Seq2SeqLabs.jsx';
export default {
 title: 'Sequence-to-Sequence & Encoder-Decoder',
 readTime: '~65 min read + experiments and practice',
 hasIntegratedGuide: true,
 content: () => <div className="neural-lesson seq2seq-lesson"><LessonIntro prerequisites="Recurrent state updates and cross-entropy. Source/target tokens, shapes, context and loss normalization are refreshed before composing the model." sections={[["1-two-timelines-one-conditional-task","1. Two timelines, one conditional task"],["2-give-every-token-and-tensor-a-job","2. Give every token and tensor a job"],["3-how-the-encoder-and-decoder-communicate","3. How the encoder and decoder communicate"],["4-train-the-probability-of-the-whole-answer","4. Train the probability of the whole answer"],["5-generate-an-answer-then-search-more-than-one-route","5. Generate an answer, then search more than one route"],["6-a-real-experiment-learning-a-function-is-harder-than-remembering-pairs","6. A real experiment: learning a function is harder than remembering pairs"],["7-run-the-complete-small-model","7. Run the complete small model"],["8-diagnose-the-model-by-separating-four-questions","8. Diagnose the model by separating four questions"],["reuse-the-cell-implement-the-encoder-decoder-protocol","Reuse the cell; implement the encoder–decoder protocol"],["9-deeper-connections-and-practical-extensions","9. Deeper connections and practical extensions"],["10-practice-build-diagnose-and-change-the-problem","10. Practice: build, diagnose and change the problem"],["11-references-another-way-to-learn-and-the-next-step","11. References, another way to learn, and the next step"]]}>Read an input, learn a conditional output and inspect the difference between memorizing examples, generating an answer and finding a better route.</LessonIntro>
<Prose>{""}<strong>{"Explore as you read."}</strong>{" Edit source/target shifts, bridge weights/rate, tiny probability trees, beam width and supported fitted source/prefix inputs. Show aligned timelines, dependency paths, sequence probabilities and bounded beam candidates live. Keep teacher-forced versus generated inputs explicit at every step. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to distinguish model probability from a decoding decision and identify when a prefix or alignment changes the actual task."}</Prose>

<Prose>{"A handwriting classifier reads several pen positions and chooses one digit. Now change the request: read a word and generate its past tense. The input "}<code>{"walk"}</code>{" has four characters; "}<code>{"walked"}</code>{" has six. "}<code>{"eat"}</code>{" becomes "}<code>{"ate"}</code>{", so copying the input and appending a suffix is not enough. The system must decide both "}<strong>{"what comes next"}</strong>{" and "}<strong>{"when the answer is finished"}</strong>{"."}</Prose>

<Prose>{"A sequence-to-sequence model learns a mapping from an ordered input to an ordered output. An "}<strong>{"encoder–decoder"}</strong>{" is one way to build it: an encoder turns the input into a learned representation; a decoder uses that representation to produce the output. Here we will build a small recurrent version, inspect its actual states, and discover why fitting the training examples does not mean it has learned a reusable spelling rule."}</Prose>

<Prose>{""}<strong>{"First pass:"}</strong>{" follow sections 1–6 to understand the two networks, the shifted training targets, generation and a real experiment. Section7 supplies the complete CPU program. Use section 8 to diagnose it, then try the practice. The deeper branches in section 9 are optional on a first reading; they explain architectural variants, training objectives and production contracts."}</Prose>

<Prose>{"You need the idea that a recurrent state is an updated vector and that cross-entropy rewards probability placed on the correct outcome. We refresh both below. "}<a href={"/learn/path/full-curriculum/rnns-lstms-grus?module=deep-learning-fundamentals"}>{"RNNs, LSTMs and GRUs"}</a>{" provides the full cell equations."}</Prose>

<H2>{"1. Two timelines, one conditional task"}</H2>

<Prose>{"Imagine a language-learning tool that is given a "}<strong>{"lemma"}</strong>{", the dictionary form of a word, and a grammatical request:"}</Prose>

<NeuralTable caption={"1. Two timelines, one conditional task"} headers={[<>{"Input"}</>,<>{"Requested form"}</>,<>{"Output"}</>]} rows={[[<>{""}<code>{"walk"}</code>{""}</>,<>{"past"}</>,<>{""}<code>{"walked"}</code>{""}</>],[<>{""}<code>{"try"}</code>{""}</>,<>{"past"}</>,<>{""}<code>{"tried"}</code>{""}</>],[<>{""}<code>{"make"}</code>{""}</>,<>{"present participle"}</>,<>{""}<code>{"making"}</code>{""}</>],[<>{""}<code>{"eat"}</code>{""}</>,<>{"past"}</>,<>{""}<code>{"ate"}</code>{""}</>]]} />

<Prose>{"The grammatical request is part of the input. Without it, the same word could legitimately require several answers. A training example therefore includes the source information and the desired output; the network cannot infer an omitted task specification merely from being large."}</Prose>

<Prose>{"Our running data example is "}<code>{"lactate + past → lactated"}</code>{", a real entry in the supplied UniMorph extract. A "}<strong>{"token"}</strong>{" is one item the network processes. In this lesson characters are tokens, and the grammatical request is a separate token:"}</Prose>

<Prose>{""}<code>{"<past> → l → a → c → t → a → t → e → <eos>"}</code>{""}</Prose>

<Prose>{"The encoder processes those nine source tokens. The decoder starts a different timeline:"}</Prose>

<Prose>{""}<code>{"<bos> → predict l → predict a → … → predict d → predict <eos>"}</code>{""}</Prose>

<Prose>{""}<code>{"<bos>"}</code>{" means “begin generating.” "}<code>{"<eos>"}</code>{" is an actual predicted outcome meaning “end the output.” The source also has its own end marker. The two end markers share an ID in our program, but occupy different sequences. Neither is a letter in the word."}</Prose>

<Seq2SeqTimelines />

<Prose>{"This is different from labeling every source token. A tagger might assign one label to each word; this decoder is free to output a different number of tokens. Translation, speech transcription with an autoregressive decoder, spelling normalization and generation of structured text can use the same broad input/output contract. Their token choices, valid outputs and error costs differ. A recurrent encoder–decoder is not the only architecture capable of these tasks."}</Prose>

<Prose>{"An interesting practical connection is "}<strong>{"inflection for language tools"}</strong>{". A dictionary assistant may need hundreds of forms of a lemma. A general learned mapping can share patterns across examples, while a rule system can directly encode predictable changes. Neither approach makes dictionary exceptions disappear. Our experiment keeps the rule system in the comparison rather than assuming the neural network should replace it."}</Prose>

<H2>{"2. Give every token and tensor a job"}</H2>

<Prose>{"The model cannot multiply the string "}<code>{"\"a\""}</code>{" by a matrix. We assign each token an integer ID, then use an "}<strong>{"embedding table"}</strong>{": a learned row vector for each ID. Looking up row 17 does not claim that token 17 is “larger” than token 8. The number is an address."}</Prose>

<Prose>{"Our vocabulary has 32 entries: 26 lowercase letters, three grammatical-request tokens, and "}<code>{"<pad>"}</code>{"/"}<code>{"<bos>"}</code>{"/"}<code>{"<eos>"}</code>{". The input contract restricts this small experiment to lowercase English spellings. A general text system must decide how it handles other scripts, case, punctuation, unknown tokens and normalization. Word, subword and byte tokenization are separate choices; encoder–decoder learning does not require BPE."}</Prose>

<Prose>{""}<code>{"<pad>"}</code>{" fills unused cells when examples of different lengths share a batch. It is storage, not a requested prediction. We use two distinct mechanisms:"}</Prose>

<ol><li>{""}<strong>{"Source lengths:"}</strong>{" packing tells the recurrent encoder which source positions exist. The final state corresponds to the last real source token, including its EOS."}</li><li>{""}<strong>{"Target mask:"}</strong>{" the loss ignores target PAD positions. Output EOS is real and remains in the loss."}</li></ol>

<Prose>{"The decoder can emit only letters or EOS in this task. Before softmax, the program masks PAD, BOS and request-token logits to negative infinity. Their output probabilities become zero. This output-support rule is applied consistently during training and generation; it is not a late cosmetic cleanup of bad strings."}</Prose>

<Prose>{"For a batch of "}<code>{"B"}</code>{" examples, longest source length "}<code>{"S"}</code>{" and longest target length "}<code>{"T"}</code>{":"}</Prose>

<NeuralTable caption={"2. Give every token and tensor a job"} headers={[<>{"Object"}</>,<>{"Shape in the program"}</>,<>{"Meaning"}</>]} rows={[[<>{"Source IDs"}</>,<>{""}<code>{"B × S"}</code>{""}</>,<>{"Request, letters, EOS, then padding"}</>],[<>{"Source embeddings"}</>,<>{""}<code>{"B × S × 24"}</code>{""}</>,<>{"Learned input vectors"}</>],[<>{"Encoder final state"}</>,<>{""}<code>{"1 × B × 64"}</code>{""}</>,<>{"One GRU layer, batch, state coordinates"}</>],[<>{"Decoder input IDs"}</>,<>{""}<code>{"B × T"}</code>{""}</>,<>{"BOS followed by the known target prefix during training"}</>],[<>{"Decoder states"}</>,<>{""}<code>{"B × T × 64"}</code>{""}</>,<>{"One state for each predicted output position"}</>],[<>{"Output logits"}</>,<>{""}<code>{"B × T × 32"}</code>{""}</>,<>{"Scores before softmax; invalid outputs masked"}</>],[<>{"Target IDs"}</>,<>{""}<code>{"B × T"}</code>{""}</>,<>{"Desired letters followed by EOS, then padding"}</>]]} />

<Prose>{"For target "}<code>{"ate"}</code>{" the alignment is:"}</Prose>

<NeuralTable caption={"2. Give every token and tensor a job"} headers={[<>{"Prediction step"}</>,<>{"1"}</>,<>{"2"}</>,<>{"3"}</>,<>{"4"}</>]} rows={[[<>{"Decoder receives"}</>,<>{"BOS"}</>,<>{"a"}</>,<>{"t"}</>,<>{"e"}</>],[<>{"Correct output"}</>,<>{"a"}</>,<>{"t"}</>,<>{"e"}</>,<>{"EOS"}</>]]} />

<Prose>{"At step 2 the decoder receives "}<code>{"a"}</code>{" because it is supposed to predict the token "}<strong>{"after"}</strong>{" "}<code>{"a"}</code>{". Feeding "}<code>{"t"}</code>{" at that same step would reveal the answer being scored. Forgetting the shift can produce an impressively small loss for the wrong task."}</Prose>

<Seq2SeqAlignmentLab />

<H2>{"3. How the encoder and decoder communicate"}</H2>

<Prose>{"Let "}<InlineMath>{"x_1,\\ldots,x_S"}</InlineMath>{" be source token IDs and "}<InlineMath>{"E[x_i]"}</InlineMath>{" their embedding vectors. The encoder updates a state:"}</Prose>

<div className="neural-equation"><MathBlock>{"h_i=\\operatorname{GRU}_{enc}(E[x_i],h_{i-1}),\\qquad h_0=0."}</MathBlock></div>

<Prose>{"Its final state "}<InlineMath>{"c=h_S"}</InlineMath>{" is the "}<strong>{"context"}</strong>{". In our basic architecture the decoder begins with "}<InlineMath>{"s_0=c"}</InlineMath>{":"}</Prose>

<div className="neural-equation"><MathBlock>{"s_t=\\operatorname{GRU}_{dec}(E[y_{t-1}],s_{t-1}),\\qquad y_0=\\mathrm{BOS},"}</MathBlock></div>

<div className="neural-equation"><MathBlock>{"z_t=W_{out}s_t+b_{out},\\qquad p_t=\\operatorname{softmax}(z_t)."}</MathBlock></div>

<Prose>{"The two GRUs have different learned parameters. “Both use a GRU” means they use the same kind of calculation, not the same weights. The embedding table happens to be shared because this task uses the same letters on both sides; separate source and target embeddings are also possible."}</Prose>

<Prose>{"The decoder does not receive the raw spelling again in this version. After initialization, all source influence must travel through its evolving state. If we replace one input's context with another's while keeping the decoder and BOS fixed, it generates from that other context. This is a useful, testable meaning of “the context conditions the output.”"}</Prose>

<H3>{"A small complete forward calculation"}</H3>

<Prose>{"To make the numbers inspectable, temporarily replace the 64-coordinate GRUs with scalar tanh updates. This is a constructed mechanism example, not the fitted inflector:"}</Prose>

<div className="neural-equation"><MathBlock>{"h_i=\\tanh(0.7x_i+0.4h_{i-1}+0.1),\\quad h_0=0,\\quad x=[0.2,0.8]."}</MathBlock></div>

<Prose>{"The first state is "}<InlineMath>{"\\tanh(0.24)=0.235496"}</InlineMath>{". The second is "}<InlineMath>{"\\tanh(0.7(0.8)+0.4(0.235496)+0.1)=0.637647"}</InlineMath>{". That second state becomes the decoder's initial state."}</Prose>

<Prose>{"Use decoder update "}<InlineMath>{"s_t=\\tanh(0.6e_{t-1}+0.5s_{t-1}+0.05)"}</InlineMath>{", with BOS embedding 0.1 and token A embedding 0.4. At each step the two output scores are "}<InlineMath>{"[s_t,-s_t]"}</InlineMath>{", for A and EOS respectively."}</Prose>

<NeuralTable caption={"A small complete forward calculation"} headers={[<>{"Step"}</>,<>{"Decoder input"}</>,<>{"New state"}</>,<>{"P(A)"}</>,<>{"P(EOS)"}</>,<>{"Desired output"}</>]} rows={[[<>{"1"}</>,<>{"BOS embedding 0.1"}</>,<>{"0.404338"}</>,<>{"0.691827"}</>,<>{"0.308173"}</>,<>{"A"}</>],[<>{"2"}</>,<>{"A embedding 0.4"}</>,<>{"0.455936"}</>,<>{"0.713383"}</>,<>{"0.286617"}</>,<>{"EOS"}</>]]} />

<Prose>{"The model starts the answer reasonably but assigns too little probability to ending it. A state can look numerically stable while its prediction is wrong. A large norm or a smooth heatmap is not evidence of understanding."}</Prose>

<Seq2SeqBridgeLab />

<H2>{"4. Train the probability of the whole answer"}</H2>

<Prose>{"For a particular output "}<InlineMath>{"y_1,\\ldots,y_T"}</InlineMath>{", including its final EOS, the autoregressive model assigns:"}</Prose>

<div className="neural-equation"><MathBlock>{"P_\\theta(y\\mid x)=\\prod_{t=1}^{T}P_\\theta(y_t\\mid y_{<t},x)."}</MathBlock></div>

<Prose>{"“Autoregressive” means a prediction depends on previous output tokens. The product is the probability of this entire route through the decoder. Taking negative logarithms turns the product into a sum:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\mathcal L_{\\text{sequence}}=-\\sum_{t=1}^{T}\\log P_\\theta(y_t\\mid y_{<t},x)."}</MathBlock></div>

<Prose>{"In the scalar example, the correct-token probabilities are 0.691827 and 0.286617. Their negative-log costs are 0.368419 and 1.249609. The mean loss is 0.809014 natural-log units, or "}<strong>{"nats"}</strong>{", per target token. EOS contributes most of the error here. Removing EOS from the targets would remove the direct lesson “finish after A.”"}</Prose>

<Prose>{"Across a padded batch, our objective is the sum of valid-token costs divided by the "}<strong>{"number of valid target tokens"}</strong>{":"}</Prose>

<div className="neural-equation"><MathBlock>{"\\mathcal L=\\frac{\\sum_{b,t}m_{bt}[-\\log p_{bt}(y_{bt})]}{\\sum_{b,t}m_{bt}},\\quad\nm_{bt}=1\\ \\text{when the target is not PAD}."}</MathBlock></div>

<Prose>{"This gives equal weight to valid tokens. Averaging each sequence first would give equal weight to sequences and therefore relatively more weight to tokens in short answers. Both can be deliberate objectives; changing the denominator silently changes the training problem."}</Prose>

<H3>{"Teacher forcing: a known prefix, not the current answer"}</H3>

<Prose>{"During training we know the reference output. "}<strong>{"Teacher forcing"}</strong>{" uses its previous tokens as the decoder inputs while scoring each next token. This directly evaluates the conditional factors in the likelihood above. It is ordinary maximum-likelihood training for this model, not a trick that makes the loss invalid."}</Prose>

<Prose>{"The GRU states still depend on earlier states. Providing all known input tokens permits a convenient batched call to "}<code>{"nn.GRU"}</code>{", but it does not make recurrent time steps mathematically independent. The reference token at step 5 cannot determine state5 without the recurrent history."}</Prose>

<Prose>{"At inference the reference answer is unavailable. The decoder must use a chosen or sampled previous output. If it makes a mistake, subsequent states may follow a prefix poorly represented in training. This is a reason to measure complete generated answers as well as teacher-forced loss. It does not imply that every initial error causes an irreversible cascade."}</Prose>

<H3>{"One gradient reaches both networks"}</H3>

<Prose>{"Backpropagation starts at the output losses, passes through decoder states and the initial context, then through the encoder. In the scalar example, hold the decoder fixed and differentiate with respect to the encoder's input weight 0.7. The calculated derivative is −0.00556958. A gradient-descent step of size 0.1 changes it to 0.70055696 and reduces mean loss to 0.80901091. The change is small, but it demonstrates the important dependency: an output loss can teach the encoder how to represent its input."}</Prose>

<Prose>{"The supplied calculation checks that derivative against a central finite difference. In the real model, a four-example batch gives a nonzero encoder input-weight gradient norm of 0.169993. Detaching the context gives the same forward computation but removes this decoder-to-encoder gradient path. That would defeat joint learning unless a separate encoder objective were intended."}</Prose>

<details>

<summary>Follow the scalar derivative through the bridge</summary>

<Prose>{"For the two-token mean loss, the direct derivative at decoder step 1 is "}<InlineMath>{"p_1(A)-1"}</InlineMath>{", and at step 2 it is "}<InlineMath>{"p_2(A)"}</InlineMath>{" because EOS is correct there. Step2 also sends credit back through step 1:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{\\partial\\mathcal L}{\\partial s_1}\n=(p_1(A)-1)+p_2(A)\\,0.5(1-s_2^2)."}</MathBlock></div>

<Prose>{"Multiply this by "}<InlineMath>{"0.5(1-s_1^2)"}</InlineMath>{" to cross the decoder-initial-state edge into context "}<InlineMath>{"c"}</InlineMath>{". The encoder input weight is reused at both source positions, giving:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{\\partial c}{\\partial w}\n=(1-h_2^2)\\left[x_2+0.4(1-h_1^2)x_1\\right]."}</MathBlock></div>

<Prose>{"The product is −0.00556958. The first term in brackets is the direct contribution at source position 2; the second comes through the earlier encoder state. Both belong to the same parameter."}</Prose>

</details>

<Prose>{"Return to "}<a href={"/learn/path/full-curriculum/backpropagation-automatic-differentiation?module=deep-learning-fundamentals"}>{"Backpropagation and Automatic Differentiation"}</a>{" for the full graph mechanics. Here the new point is where the two graphs meet."}</Prose>

<H2>{"5. Generate an answer, then search more than one route"}</H2>

<Prose>{""}<strong>{"Greedy decoding"}</strong>{" chooses the largest next-token probability. It starts from BOS, updates the state, emits one token, and feeds that token into the next step. It stops on EOS or an explicit output limit. Hitting the limit means “generation was capped,” not “the model chose to finish.”"}</Prose>

<Prose>{"For our fitted seed 1 model, the training input "}<code>{"lactate + past"}</code>{" generates "}<code>{"lactated"}</code>{" and EOS. Each emitted token has a probability conditional on that particular input and generated prefix. Multiplying those probabilities does not produce “probability this spelling is linguistically correct.” It is the model's probability for that route."}</Prose>

<H3>{"Why the locally best token can lose"}</H3>

<Prose>{"Consider this completely specified constructed tree. First choose A with probability 0.60 or B with 0.40. After A, choose EOS with 0.51 or C with 0.49. After B, choose EOS with 0.90 or C with 0.10. After C, EOS is mandatory."}</Prose>

<NeuralTable caption={"Why the locally best token can lose"} headers={[<>{"Complete answer"}</>,<>{"Product"}</>,<>{"Probability"}</>]} rows={[[<>{"A, EOS"}</>,<>{"0.60 × 0.51"}</>,<>{"0.306"}</>],[<>{"A, C, EOS"}</>,<>{"0.60 × 0.49 × 1"}</>,<>{"0.294"}</>],[<>{"B, EOS"}</>,<>{"0.40 × 0.90"}</>,<>{"0.360"}</>],[<>{"B, C, EOS"}</>,<>{"0.40 × 0.10 × 1"}</>,<>{"0.040"}</>]]} />

<Prose>{"Greedy commits to A, then EOS: 0.306. The highest-probability complete answer is B, EOS: 0.360. The first decision can look best until we consider what follows it."}</Prose>

<Prose>{""}<strong>{"Beam search"}</strong>{" retains several partial answers. With width 2, keep A and B after the first step. Expand each live candidate, add its next-token log probability to its accumulated log score, then retain the two highest-scoring candidates. In this tree those are B,EOS and A,EOS. A finished candidate is retained without appending more tokens."}</Prose>

<Prose>{"Each candidate owns its prefix "}<strong>{"and its decoder state"}</strong>{". Reusing the state from the wrong candidate makes the next distribution wrong even if the displayed strings look right. A beam is not simply a list of alternative final words from one shared state."}</Prose>

<Prose>{"Our program keeps completed and live candidates in the same width-limited list, sorts tied scores by token IDs, and stops when every retained candidate is complete or 16 generated tokens have been reached. At a cap it returns the highest-ranked remaining candidate with an explicit termination flag. Width1 matches the program's greedy algorithm under the same support, tie order and stopping convention."}</Prose>

<Prose>{"The search remains approximate: a promising route can be pruned before its good continuation appears. A larger beam also optimizes the model's score more thoroughly, which need not improve the task metric when the model is wrong. The real experiment below measures the effect instead of claiming a guaranteed improvement."}</Prose>

<H3>{"Length normalization changes the objective"}</H3>

<Prose>{"Raw log probability is a sum of nonpositive terms. Extending a particular prefix cannot increase that raw probability. Comparing completed answers of different lengths can therefore require an explicit length policy."}</Prose>

<Prose>{"One common score is:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\operatorname{score}(y)=\\frac{\\log P(y\\mid x)}\n{\\left((5+L)/6\\right)^\\alpha},"}</MathBlock></div>

<Prose>{"where our definition of "}<InlineMath>{"L"}</InlineMath>{" counts generated tokens including EOS and excludes BOS. This is the length term from the GNMT family of scoring rules; its separate coverage term requires attention and is not used here. The exponent is applied "}<strong>{"once"}</strong>{". It is a ranking heuristic, not a normalized probability distribution. "}<a href={"https://arxiv.org/abs/1609.08144"}>{"GNMT, section 7"}</a>{"."}</Prose>

<Prose>{"For a length 3 answer with log probability−1.5 and a length 6 answer with −1.8, raw scoring prefers the shorter answer. At "}<InlineMath>{"\\alpha=1"}</InlineMath>{", their scores are −1.125 and−0.981818, so the longer answer wins. Dividing a negative number by a larger positive denominator makes it less negative. Calling this universally a penalty against long outputs reverses the actual effect."}</Prose>

<Prose>{"The experiment fixes "}<InlineMath>{"\\alpha=0"}</InlineMath>{". A nonzero setting must be chosen on development data and reported with the length convention. Do not borrow a number from a translation paper as a universal spelling-model setting."}</Prose>

<Seq2SeqTreeLab />

<H2>{"6. A real experiment: learning a function is harder than remembering pairs"}</H2>

<Prose>{"The offline extract comes from the "}<a href={"https://github.com/unimorph/eng"}>{"UniMorph English repository"}</a>{", pinned to a specific source revision. It contains 600 lowercase lemma spellings, each with a past, present-participle and third-person-singular-present form. The source includes uncommon and historical spellings; it is a lexicon sample, not the frequency distribution of everyday English. Its data license and attribution travel with the extract."}</Prose>

<Prose>{"The intended question is: "}<strong>{"can this model inflect a spelling absent from training?"}</strong>{" All three requests for one lemma stay together. We also group selected lemmas linked by a shared output form: "}<code>{"worke"}</code>{" and "}<code>{"work"}</code>{" share "}<code>{"worked"}</code>{" and "}<code>{"working"}</code>{". Treating their strings as unrelated would give a misleadingly clean “no overlap” report. This conservative grouping can merge genuine homographs; it is an operational split policy, not a complete linguistic identity system."}</Prose>

<Prose>{"The final split has 451 training lemmas/1,353 examples and 149 development lemmas/447 examples. There is no shared lemma string or target form across these partitions. No unseen final test is claimed: we inspect development outcomes to learn about the system. Future tuning would consume more development information and require a separately protected evaluation for a final performance claim."}</Prose>

<Prose>{"The experiment fixes a 24-coordinate embedding, one 64-coordinate GRU encoder, one 64-coordinate GRU decoder and a 32-score output head: 37,408 trainable parameters. Each of three seeds receives 1,200 Adam updates at learning rate 0.003, batches of 64 sampled training rows, global gradient clipping at 1, teacher forcing and the same token/split rules. We show the final predeclared update, not the best-looking development checkpoint."}</Prose>

<Prose>{"The comparison includes two non-neural systems. "}<strong>{"Copy"}</strong>{" returns the lemma unchanged. "}<strong>{"Predeclared suffix rules"}</strong>{" append or replace endings such as "}<code>{"y→ied"}</code>{" and "}<code>{"e→ing"}</code>{"; the exact rules are in the program. They omit irregular dictionaries and some spelling conditions, including consonant doubling. Their prior knowledge is explicit."}</Prose>

<Prose>{"For generated strings, "}<strong>{"exact match"}</strong>{" requires the reference spelling and natural EOS termination. "}<strong>{"Character error rate"}</strong>{" is total Levenshtein insertions, deletions and substitutions divided by total reference characters. A rate above 1 is possible when a generator inserts many characters. Teacher-forced NLL is measured separately, including EOS and excluding PAD."}</Prose>

<NeuralTable caption={"6. A real experiment: learning a function is harder than remembering pairs"} headers={[<>{"System"}</>,<>{"Training exact / 1,353"}</>,<>{"Development exact / 447"}</>,<>{"Development character error rate"}</>,<>{"Development teacher-forced NLL"}</>]} rows={[[<>{"Copy lemma"}</>,<>{"3"}</>,<>{"1"}</>,<>{"0.255014"}</>,<>{"Not a probabilistic model"}</>],[<>{"Predeclared suffix rules"}</>,<>{"1206"}</>,<>{"407"}</>,<>{"0.016332"}</>,<>{"Not a probabilistic model"}</>],[<>{"GRU encoder–decoder, seed 1"}</>,<>{"1328"}</>,<>{"53"}</>,<>{"0.434670"}</>,<>{"1.285388"}</>],[<>{"GRU encoder–decoder, seed 2"}</>,<>{"1323"}</>,<>{"41"}</>,<>{"0.439255"}</>,<>{"1.455556"}</>],[<>{"GRU encoder–decoder, seed 3"}</>,<>{"1287"}</>,<>{"52"}</>,<>{"0.422636"}</>,<>{"1.232162"}</>]]} />

<Prose>{"All final neural development generations ended with EOS within the limit. These are actual CPU measurements on the declared grouped split."}</Prose>

<Prose>{"The neural system can memorize training pairs yet fail badly on new lemmas. Most of this task's characters should be copied from the input. A fixed context forces the model to learn a representation and a decoder that preserve this detail; the hand-written rules already preserve it by construction. The comparison is informative even though the neural model loses."}</Prose>

<Prose>{"Do not conclude that all encoder–decoders are poor inflectors. This experiment has a small training lexicon, a particular architecture, initialization, optimizer and budget. It establishes this result for this protocol. Increasing capacity alone might improve memorization without solving the held-out problem."}</Prose>

<Prose>{"Seed 1's development slices are 44/168 exact for lemma lengths 3–5 and 9/279 for lengths 6–8. Longer spellings are harder here, but length is entangled with which words occur, spelling patterns and number of required copy operations. This is not a causal experiment proving that a fixed context has a universal character limit."}</Prose>

<Prose>{"The same seed's beam 3 search returns 56/447 exact rather than greedy's 53/447; character error rate changes from 0.434670 to 0.410888. No additional training occurred. Search recovers some better-scoring routes, but it cannot supply missing linguistic knowledge or undo the large generalization gap."}</Prose>

<Seq2SeqEvidenceFigure />

<H2>{"7. Run the complete small model"}</H2>

<Prose>{"Save "}<code>{"english-inflections.csv"}</code>{" beside "}<code>{"inflection-seq2seq.py"}</code>{". A CPU Python environment with NumPy and PyTorch is sufficient; no pretrained model or runtime data download is used. The recorded run used Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu. If those packages are absent, create a local virtual environment and install NumPy and the CPU build of PyTorch appropriate to your operating system from its official installer."}</Prose>

<Prose>Download the <a href={seq2seqAsset + "english-inflections.csv"}>offline inflection CSV</a>, <a href={seq2seqAsset + "inflection-seq2seq.py"}>complete program</a> and <a href={seq2seqAsset + "data-provenance.md"}>attribution and split provenance</a>. Preserve the UniMorph English contributors, source revision, adaptation and CC BY-SA 3.0 license when sharing the extract. To inspect the fixed model without rerunning training, also save <a href={seq2seqAsset + "calculated-inputs.json"}>the measured record and weights</a> plus <a href={seq2seqAsset + "saved-inflection.py"}>the standalone inference example</a>.</Prose>

<Prose>{"Run "}<code>{"python inflection-seq2seq.py"}</code>{". The program prints checkpoint and final metrics and writes "}<code>{"calculated-inputs.json"}</code>{" with predictions, weights and the protocol. Seeds make the experiment reproducible in the recorded environment; another backend/version may produce small numerical differences."}</Prose>

<Prose>{"Read the program in this order: "}<code>{"batch"}</code>{" constructs the shifted tracks; "}<code>{"Inflector"}</code>{" connects the states; "}<code>{"greedy"}</code>{" performs free generation; "}<code>{"beam"}</code>{" owns each candidate state; "}<code>{"summarize"}</code>{" separates exact match, edits and termination; "}<code>{"main"}</code>{" applies the fixed training protocol. "}<code>{"assess"}</code>{" uses no gradients. "}<code>{"model.eval()"}</code>{" chooses evaluation behavior, whereas "}<code>{"no_grad()"}</code>{" suppresses gradient recording."}</Prose>

<Seq2SeqProgram file="inflection-seq2seq.py" title="Read the complete runnable inflection model" />

<Prose>{"The decoder's previous token is selected with "}<code>{"argmax"}</code>{" only during generation. The training graph uses the known target prefix and differentiable logits. We do not backpropagate through the discrete greedy choices to fit this maximum-likelihood model."}</Prose>

<Prose>{"The fixed learned model also supports inference without fitting. After the first run, save this next block beside the program and JSON and run it. It loads seed1's saved parameters, asks only for the lemma and grammatical request, and prints both the string and whether EOS ended it. No reference answer is supplied to generation."}</Prose>

<CodeBlock language={"python"}>{"from pathlib import Path\nimport json\nimport runpy\nimport torch\n\nfolder = Path(__file__).resolve().parent\napi = runpy.run_path(str(folder / \"inflection-seq2seq.py\"))\nsaved = json.loads((folder / \"calculated-inputs.json\").read_text())\nmodel = api[\"Inflector\"]().eval()\nweights = saved[\"runs\"][0][\"weights\"]\nmodel.load_state_dict({\n    key: torch.tensor(value, dtype=torch.bool if key == \"invalid_output\" else torch.float32)\n    for key, value in weights.items()\n})\nquery = {\"lemma\": \"lactate\", \"feature\": \"past\"}\nfor limit in (3, 16):\n    result = api[\"greedy\"](model, [query], max_output=limit)[0]\n    print(limit, result[\"prediction\"], result[\"ended_with_eos\"])"}</CodeBlock>

<Prose>{"The recorded output is "}<code>{"3 lac False"}</code>{" followed by "}<code>{"16 lactated True"}</code>{" for the worked training example. The browser investigations will similarly load one saved seed only when needed and perform bounded inference on short inputs, not train 1,200 updates in a page."}</Prose>

<H2>{"8. Diagnose the model by separating four questions"}</H2>

<Prose>{""}<strong>{"Did we ask and encode the right task?"}</strong>{" Inspect the request token, source lengths, target shift, EOS and output vocabulary. Swapping a row's reference form must not change the encoder input. If a source spelling is edited, rerun the encoder; do not keep its old context. A typo or unsupported character needs an explicit input decision."}</Prose>

<Prose>{""}<strong>{"Can the network fit the supplied examples?"}</strong>{" Low training loss and many exact training outputs show optimization has learned something about those pairs. If even a small training batch cannot be learned, inspect gradients, state detachment, masking, data pairing and parameter updates before adding decoding restrictions."}</Prose>

<Prose>{""}<strong>{"Does the learned rule transfer?"}</strong>{" Compare the same fixed model on grouped development examples and the simple baseline. Examine actual outputs, not only a scalar average. A grammatically plausible suffix attached to the wrong copied stem is still an error. A reference may itself contain an unusual lexicon entry; inspect provenance before “correcting” it to your expectation."}</Prose>

<Prose>{""}<strong>{"Is search failing to find a good route the model already scores well?"}</strong>{" Compare greedy and beam under a fixed model and declared scoring rule. An increase in model score with a decrease in exact match is possible. Beam size is an inference choice; training updates are a model change. Mixing them in one unexplained curve hides what caused the result."}</Prose>

<Seq2SeqFittedLab />

<Prose>{"A useful null experiment replaces the context with an exact copy of itself: nothing should change. Replaying an already generated prefix should recover the same continuation. Replacing it with a different source's context should use that context's information. Changing only an on-screen label should affect none of the numbers."}</Prose>

<Prose>{"Do not ban repeated characters merely because a model repeats. "}<code>{"letter"}</code>{" and "}<code>{"unsubbed"}</code>{" legitimately contain repeats. Diagnose data and generation first; use a constraint only when the task itself rules out the affected outputs."}</Prose>

<H2>{"Reuse the cell; implement the encoder–decoder protocol"}</H2>

<Prose>{"This lesson's new program is the protocol joining two recurrent computations, not another invention of GRU. "}<a href={"/learn-code/sequence-to-sequence-encoder-decoder/sequence-mechanics.py"}>{"sequence-mechanics.py"}</a>{" opens the scalar joint derivative, manual gate trace and exact small probability-tree search. "}<a href={"/learn-code/sequence-to-sequence-encoder-decoder/inflection-seq2seq.py"}>{"inflection-seq2seq.py"}</a>{" supplies complete source batching, "}<code>{"Inflector"}</code>{", training, greedy decoding, beam search and evaluation. The "}<a href={"/learn-assets/rnns-lstms-grus/recurrent-mechanics.py"}>{"recurrent cell implementation"}</a>{" maps gate order and biases to "}<code>{"nn.GRU"}</code>{". Its "}<a href={"/learn/path/full-curriculum/rnns-lstms-grus?module=deep-learning-fundamentals"}>{"RNN, LSTM and GRU lesson"}</a>{" teaches the scratch cell and matched library route; reuse that mechanism here."}</Prose>

<Seq2SeqProgram file="sequence-mechanics.py" title="Read the scalar joint derivative, manual GRU protocol and exact search checks" />

<Prose>{"The ordinary implementation uses embeddings and "}<code>{"nn.GRU"}</code>{" for encoding/decoding, and explicit code for shifting targets, carrying context and deciding when to end. There is no requirement to replace this small research model with a downloaded language-model wrapper. The supplied beam function owns candidate state: token IDs, accumulated log probability, end status and decoder state must travel together. A batched decoder can share the encoder memory, but its beam-specific hidden states cannot be accidentally shared and mutated."}</Prose>

<Prose>{"The hand-search tree and trained inflector answer different questions. The tree checks search arithmetic exactly; the trained model checks whether learned conditional distributions support useful outputs. Width1 beam should match greedy under the same tie/termination rule. An ended hypothesis is retained without repeatedly consuming EOS, while a hypothesis that reaches the step cap is reported as capped. Length normalization changes ranking; it is not a harmless numerical rescaling."}</Prose>

<Prose>{""}<strong>{"Changed-code task:"}</strong>{" add a second source to a batched decoding routine, one ending after2 tokens and another after5. Keep an explicit ended mask and original source IDs. After an example ends, preserve its final sequence and stop assigning it new scored tokens; continue the other example. Test that decoding this batch gives the same two results as separate calls in eval mode. For beam search additionally reorder decoder states with the same parent indices used to gather candidate tokens."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"A batch is a collection of independent sequence states, not one common EOS event."}</Prose>

</details>

<details>

<summary>Solution and success criteria</summary>

<Prose>{"Initialize one hidden state and ended flag per source. On each step form candidate logits only for active rows, append their chosen tokens, mark newly emitted EOS and gather any beam parents consistently. Already-ended output strings remain unchanged. Compare complete token sequences and log probabilities, including the case where one row is capped and the other genuinely ended. Padding is storage, not another generated token. Correct source-to-state ownership matters more than saving a few Python lines."}</Prose>

</details>

<H2>{"9. Deeper connections and practical extensions"}</H2>

<details>

<summary>Different context interfaces: GRU, LSTM, stacks and attention</summary>

<Prose>{"Our decoder initializes from one GRU state. An LSTM has both hidden state "}<InlineMath>{"h"}</InlineMath>{" and cell state "}<InlineMath>{"c"}</InlineMath>{"; a complete handoff must say what happens to both. With several layers, state has a layer axis. A bidirectional encoder has two directional states per layer. “Pass the final hidden vector” is insufficient when the decoder expects a different shape or a missing cell state."}</Prose>

<Prose>{"If encoder and decoder widths differ, a learned projection can map a concatenated encoder representation into the required decoder state. For example, two encoder directions of 64 coordinates can be concatenated into 128 and projected to a 64-coordinate decoder state. An LSTM may need separate projections for hidden and cell states. A deliberate zero initialization with a separately supplied context is another design."}</Prose>

<Prose>{"Context can also be concatenated to the decoder input at every step. That repeatedly supplies the same source summary, whereas attention supplies a "}<strong>{"different weighted combination of source states"}</strong>{" for each decoding step. The next lesson derives that mechanism. The difference is access to information, not a guarantee that the weights perfectly explain language or that generation becomes factual."}</Prose>

<Prose>{"The 2014 Sutskever system demonstrated large recurrent encoder–decoder translation with word vocabularies and unknown-word tokens. Its reported 34.81 BLEU result used an ensemble of five models and beam 12. Source reversal shortened some important dependency paths; it did not reverse the target language or eliminate recurrent computation. The paper actually reported good performance on long sentences in that setting, so a universal “fails after 30 words” claim would misrepresent it. "}<a href={"https://arxiv.org/abs/1409.3215"}>{"Sequence to Sequence Learning with Neural Networks"}</a>{"."}</Prose>

</details>

<details>

<summary>Teacher forcing, scheduled sampling and sequence objectives</summary>

<Prose>{"Maximum likelihood scores observed prefixes. Deployment uses generated prefixes. That mismatch can expose weaknesses in an imperfect model, but the likelihood objective remains mathematically coherent. A generated-prefix intervention measures a particular response; it does not prove a universal account of every sequence error."}</Prose>

<Prose>{"Scheduled sampling mixes reference and generated previous tokens during training, typically changing the mixing probability over time. The targets can remain the original next tokens even when the prefix has changed. That means it is no longer simply evaluating the original data likelihood. The original proposal reported useful results, while an analysis of the sampling objective showed an inconsistency even in a two-symbol setting: replacing the first symbol independently can encourage prediction of the second marginal rather than the correct conditional relationship. It is not an automatic required upgrade. "}<a href={"https://arxiv.org/abs/1506.03099"}>{"Bengio et al."}</a>{", "}<a href={"https://arxiv.org/abs/1511.05101"}>{"Huszár's analysis, section 4"}</a>{"."}</Prose>

<Prose>{"Sequence-level objectives can optimize a reward or risk attached to the complete answer. They introduce their own estimation, optimization and evaluation questions. Label smoothing changes target distributions at the loss; it does not itself train on the model's wrong prefixes. Keep these mechanisms distinct when interpreting an experiment."}</Prose>

</details>

<details>

<summary>Search costs, stopping and model deployment</summary>

<Prose>{"With output vocabulary size "}<InlineMath>{"V"}</InlineMath>{", limit "}<InlineMath>{"T"}</InlineMath>{" and beam width "}<InlineMath>{"K"}</InlineMath>{", exhaustive enumeration has exponentially many possible paths, whereas beam expansion considers roughly "}<InlineMath>{"KVT"}</InlineMath>{" token extensions. This count omits the cost of each neural state update, embedding lookup, projection and sorting. It is an algorithmic description, not a measured latency claim."}</Prose>

<Prose>{"For raw log scores, extending a particular live path cannot improve its score. A completed candidate that already beats every live prefix cannot be beaten by descendants of those retained prefixes. This does not recover routes already pruned. Length-normalized scores need a compatible bound because their denominator changes; borrowing a raw-score stopping proof would be invalid."}</Prose>

<Prose>{"In a deployed model, record the exact tokenizer and vocabulary, source normalization, checkpoint revision, input limit, decoder-start and EOS IDs, padding side, precision/device, beam or sampling settings, score definition, output limit and termination reason. For multilingual models a language token is checkpoint-specific, not a universal string format. Keep model and tokenizer versions paired."}</Prose>

<Prose>{"An off-the-shelf "}<code>{"generate"}</code>{" API can manage these steps but does not remove their meaning. For example, current Transformers distinguishes beam stopping based on enough completed candidates, a heuristic, or a stricter search condition. Its length-penalty convention need not be the exact GNMT denominator used above. Inspect the actual configuration and documentation instead of assuming a familiar parameter name has one universal definition. "}<a href={"https://huggingface.co/docs/transformers/en/main_classes/text_generation"}>{"Transformers generation configuration"}</a>{"."}</Prose>

<Prose>{"Translation evaluation usually needs more than exact string match because multiple translations can be acceptable. BLEU measures a form of reference n-gram agreement with length handling; it is not a probability of truth. Report tokenization and metric configuration and include appropriate human/task checks. For short word forms here, exact match, character edits, accepted-reference policy and termination are easier to interpret. Speech and structured-output tasks need their own units and validity checks."}</Prose>

<Prose>{"An encoder–decoder can use source context and still invent unsupported content. A decoder-only model can also condition on an input prefix. Architecture family alone establishes neither faithfulness nor a universal speed ranking."}</Prose>

</details>

<H2>{"10. Practice: build, diagnose and change the problem"}</H2>

<H3>{"1. Repair a shifted target"}</H3>

<Prose>{"For "}<code>{"try + past → tried"}</code>{", write the six decoder inputs and six target tokens. Which target position is lost if you train only on the five letters?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"The first input starts generation, and the final output teaches termination. Each other input is the immediately preceding target."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"Inputs: BOS,t,r,i,e,d. Targets: t,r,i,e,d,EOS. Omitting the sixth target removes the supervised instruction to stop after "}<code>{"d"}</code>{". EOS is not padding."}</Prose>

</details>

<H3>{"2. Compute a fresh likelihood and loss mask"}</H3>

<Prose>{"An answer "}<code>{"go"}</code>{" followed by EOS receives correct-token probabilities 0.8,0.5,0.25. Compute its sequence probability and mean token NLL. Two padded storage positions are appended. Should the valid-token mean change?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Multiply probabilities for the route; add their negative natural logarithms for the loss. Count EOS but do not count storage padding."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"The probability is 0.1. The total NLL is "}<InlineMath>{"-\\log(0.1)=2.302585"}</InlineMath>{", so the three-token mean is 0.767528. Correctly ignored padding leaves it unchanged. Dividing the same loss sum by five instead gives 0.460517 and silently changes the scale."}</Prose>

</details>

<H3>{"3. Separate a source edit from an answer edit"}</H3>

<Prose>{"In teacher-forced training, change only the last character of a reference form. Must the encoder state change? Must the distribution predicting that changed target position change? What about the following decoder step?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Trace which array each component reads. A target being scored is not yet the previous token supplied to the decoder."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"The encoder state stays fixed because its source is unchanged. The distribution at the edited target position stays fixed if the preceding prefix is unchanged; the correct label and loss can change. The following step receives the edited character as input and can have a different state and distribution."}</Prose>

</details>

<H3>{"4. Make greedy lose, then make it win"}</H3>

<Prose>{"Use a new tree: first A 0.55/B 0.45; after A choose EOS 0.60/C 0.40; after B choose EOS 0.85/C 0.15; after C emit EOS with probability 1. Enumerate all complete paths and compare greedy with beam 2. Then change only P(EOS|A) to 0.90 and its complement accordingly."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"The first token's probability alone does not rank complete paths. Keep each conditional row normalized after the edit."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"Initially A,EOS=.33; A,C,EOS=.22; B,EOS=.3825; B,C,EOS=.0675. Greedy returns A,EOS, while beam 2 finds B,EOS. After the edit A,EOS=.495 and A,C,EOS=.055, so both return A,EOS. Improving search does not require it to return a different answer on every input."}</Prose>

</details>

<H3>{"5. Spot the leaked evaluation unit"}</H3>

<Prose>{"A dataset contains "}<code>{"worke→worked"}</code>{" in training and "}<code>{"work→worked"}</code>{" in development. A report says “all lemma strings are distinct, therefore this measures completely new lexical items.” What is wrong, and what did this lesson do?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"A string identity check is useful but narrower than a claim about linguistic identity. Inspect related variants and shared forms."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"Distinct spellings can represent closely related variants and share targets. The report overstates what its check proves. This packet conservatively links selected lemma spellings sharing a target, keeps the connected group in one partition and records the policy. It still does not claim to have solved all linguistic alias detection."}</Prose>

</details>

<H3>{"6. Explain a smaller loss but worse product"}</H3>

<Prose>{"Suppose model A has lower teacher-forced NLL, model B has better generated exact match, and a rule system beats both on this task. Which should you report? Does the discrepancy mean the NLL calculation is broken?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Each measurement asks a different question: probability on known-prefix targets, success of a generation procedure, and utility of a specific alternative."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"Report all relevant measurements with the same data split and protocol. Lower NLL can improve average probabilities without changing argmax decisions in the same way, and generated prefixes can differ from reference prefixes. The discrepancy does not itself show a broken loss. Choose according to the deployment requirements and reliable evaluation; the rule system remains a legitimate candidate."}</Prose>

</details>

<H3>{"7. Investigate a cap without pretending the answer ended"}</H3>

<Prose>{"Use the saved seed 1 model on a development input. Set the generation limit to 3 tokens and compare with 16. Record the emitted tokens, EOS flag and log score. Explain why the three-token prefix can have a higher raw score but be an incomplete answer."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"A prefix probability sums over possible future continuations; it has not yet paid the probability cost of choosing one of them and ending."}</Prose>

</details>

<details>

<summary>Solution and expected check</summary>

<Prose>{"On "}<code>{"emmove + past"}</code>{" the limit 3 output is "}<code>{"emo"}</code>{" with EOS=false and log score −0.833265. The limit 16 run continues to "}<code>{"emoves"}</code>{" and EOS in this saved model. A prefix can have a larger probability than any single complete extension. The cap flag must remain visible; do not relabel "}<code>{"emo"}</code>{" as a natural completed prediction."}</Prose>

</details>

<H3>{"8. Design the next controlled comparison"}</H3>

<Prose>{"You want to replace the single context with access to all encoder states. Name what you would keep fixed, what you would measure, and one reason an improvement would not prove that attention alone caused every difference."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Think about data groups, token support, training budget, parameter counts, decoding, seeds and what information has already been inspected."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"Keep the exact data/split, tokenization, target masking, training and evaluation definitions, decoding convention and declared seeds fixed where possible. Report parameter-count and compute changes, generated exact match, edits, termination and teacher-forced loss. Compare failures and source-length slices without selecting only favorable examples. Adding attention changes parameters and optimization as well as information access; a small controlled example is evidence for its protocol, not a universal causal ranking. The same development set remains development."}</Prose>

</details>

<H2>{"11. References, another way to learn, and the next step"}</H2>

<ul><li>{""}<a href={"https://d2l.ai/chapter_recurrent-modern/seq2seq.html"}>{"Dive into Deep Learning 1.0.3: encoder–decoder, seq2seq and beam search"}</a>{". A useful second implementation route, especially the shifted-target and masking sections. Read the beam chapter with its exact score convention in view; increasing the denominator of a negative log score does not universally penalize longer outputs."}</li><li>{""}<a href={"https://arxiv.org/abs/1409.3215"}>{"Sutskever, Vinyals and Le 2014"}</a>{". Read section 2 for the original model contract and sections 3.2–3.3 for search and source reversal. Its large translation experiment is historical evidence, not the setup of our small character model."}</li><li>{""}<a href={"https://arxiv.org/abs/1406.1078"}>{"Cho et al. 2014: RNN Encoder–Decoder"}</a>{". A complementary formulation in which source context enters the conditional decoder. Useful after you can trace our simpler initial-state interface."}</li><li>{""}<a href={"https://www.youtube.com/watch?v=XXtpJxZBa2c"}>{"Stanford CS224N 2019, Lecture 8: Translation, Seq2Seq, Attention"}</a>{", with "}<a href={"https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes06-NMT_seq2seq_attention.pdf"}>{"companion notes"}</a>{". A lecture-based alternative covering the motivation and the transition to attention. The basic encoder–decoder portions fit this lesson; return to the attention portion after the next one. The resource's age matters for “current standard” statements and framework code."}</li><li>{""}<a href={"https://unimorph.github.io/"}>{"UniMorph schema and data project"}</a>{", "}<a href={"https://aclanthology.org/2022.lrec-1.89/"}>{"UniMorph 4.0 paper"}</a>{", and "}<a href={"https://github.com/unimorph/eng/tree/66e0e9e8e2dcd196da081a25a48e5c1fe3d8b49b"}>{"the pinned English source"}</a>{". These explain what the real lexical records mean. The supplied extract retains source row numbers, filtering, grouping and CC BY-SA 3.0 attribution."}</li><li>{""}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html"}>{"PyTorch 2.14 CrossEntropyLoss"}</a>{". Use this to verify raw-logit input, class indices, "}<code>{"ignore_index"}</code>{" and reduction when adapting the program."}</li></ul>

<Prose>{"You can now connect source tokens, encoder state, decoder state, next-token probabilities, sequence loss and a complete generation procedure. Next, "}<a href={"/learn/path/full-curriculum/attention-mechanism-bahdanau-luong?module=deep-learning-fundamentals"}>{"Attention Mechanisms: Bahdanau and Luong"}</a>{" lets the decoder consult the sequence of encoder states at each output step. We will test that change on this same bounded task rather than assume it solves every failure."}</Prose>
 </div>,
};
