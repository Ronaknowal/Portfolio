// Generated from the active concept-intuition manuscript; canonical numerical programs retained.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { AttentionMemoryShelf, AttentionWorkedRead, AttentionReadLab, AttentionSchedules, AttentionFittedLab, AttentionGradientFlow, AttentionLearningCurves, AttentionProgram, AttentionWorkedAlignment, AttentionWindowLab, AttentionCopyFlow, AttentionCancellationLab, AttentionAmbiguityFigure, AttentionLocationFlow, AttentionScratchRoute } from '../../components/lesson-labs/RecurrentAttentionLabs.jsx';
import { AttentionRecallContrast, AttentionLookupBridge, AttentionSoftmaxSteps, AttentionAdditiveSteps, AttentionPaddingShares, AttentionLearningDirection } from '../../components/lesson-labs/RecurrentAttentionIntuition.jsx';
export default {
  title: 'Attention Mechanism (Bahdanau, Luong)',
  readTime: '~65 min read + 90 min experiments and practice',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson attention-lesson"><LessonIntro prerequisites="Recurrent state updates, encoder–decoder generation and cross-entropy. Queries, vector reads, shapes and masks are introduced here." sections={[["1-the-problem-one-summary-has-to-serve-every-future-question","1. The problem: one summary has to serve every future question"],["2-build-one-read-before-building-the-whole-network","2. Build one read before building the whole network"],["3-bahdanau-and-luong-distinguish-the-score-from-the-schedule","3. Bahdanau and Luong: distinguish the score from the schedule"],["4-put-the-read-into-tensors-and-code","4. Put the read into tensors and code"],["5-learn-which-read-helps-the-output","5. Learn which read helps the output"],["6-does-revisiting-memory-help-on-real-spellings","6. Does revisiting memory help on real spellings?"],["7-connect-a-real-read-to-a-generated-character","7. Connect a real read to a generated character"],["implement-the-read-then-compose-it-into-ordinary-training","Implement the read, then compose it into ordinary training"],["8-deeper-branches-other-reads-and-other-tasks","8. Deeper branches: other reads and other tasks"],["9-practice-calculate-diagnose-and-transfer","9. Practice: calculate, diagnose and transfer"],["10-continue-the-route-and-choose-another-explanation","10. Continue the route and choose another explanation"]]}>Let each output read the input it needs. Follow an exact memory read, build it from scratch, and inspect real trained spelling models.</LessonIntro>
<Prose>{"Imagine changing "}<code>{"walk"}</code>{" into its past tense, "}<code>{"walked"}</code>{", one character at a time. While writing "}<code>{"wal"}</code>{", you need the spelling. After "}<code>{"walk"}</code>{", you need to decide what ending belongs to the requested form. The useful part of the input changes as the answer grows."}</Prose>

<Prose>{"The "}<a href={"/learn/path/full-curriculum/sequence-to-sequence-encoder-decoder?module=deep-learning-fundamentals"}>{"previous encoder–decoder lesson"}</a>{" gave the writer a summary of the input. That can work. But it asks one fixed-size representation to carry everything that any later output might need. What if the writer could return to its reading notes instead?"}</Prose>

<Prose>{""}<strong>{"Attention makes that return possible."}</strong>{" At each output step, the model compares its current need with saved input representations, assigns each a share of the read, and combines the information using those shares. It can read differently when its next task changes. We will build that operation before naming its architectural variants."}</Prose>

<Prose opening="route">{""}<strong>{"First pass:"}</strong>{" follow sections 1–2 to understand one read, section 3 to place it inside a decoder, and section 4 to implement it. In section 5, read the direction-of-learning explanation before the optional derivative. Then inspect the real spelling experiment and its alignment in sections 6–7; run the saved model before a full training run. Exercises 1–5 test that route. The scorer diagnosis and section 8 are deeper branches to return to for local windows, copying, speech and efficiency."}</Prose>

<H2>{"1. The problem: one summary has to serve every future question"}</H2>

<Prose>{"A reader and a writer have different jobs. The "}<strong>{"encoder"}</strong>{" reads the source. Its state after each token is a vector: a list of learned numerical features summarizing what it has seen. The "}<strong>{"decoder"}</strong>{" writes the answer. Its own changing state reflects the output prefix and the information it has received so far."}</Prose>

<Prose>{"With a fixed-context encoder–decoder, information from the source reaches the writer through the final encoder state. A later output cannot directly request an earlier encoder state that was discarded. Attention changes this wiring: save the intermediate states, and let each writing step read a mixture of them."}</Prose>

<AttentionRecallContrast />

<Prose>{"Consider two moments in our example. With output prefix "}<code>{"wal"}</code>{", source information about the next letter can help produce "}<code>{"k"}</code>{". With prefix "}<code>{"walk"}</code>{", information about the requested tense can help produce "}<code>{"e"}</code>{". These are useful behaviors we want the network to learn, not handwritten rules telling it which position to select. Training supplies correct output strings, not a manually assigned attention location for each character."}</Prose>

<H3>{"What exactly do we keep?"}</H3>

<Prose>{"For the measured example later, the source is "}<code>{"<past> lactate <eos>"}</code>{". The word is less familiar, but the operation is the same: turn a requested spelling into "}<code>{"lactated"}</code>{". EOS means “end of sequence.” Number the source positions from 0. Save the encoder state at each position and call it "}<InlineMath>{"h_j"}</InlineMath>{", where "}<InlineMath>{"j"}</InlineMath>{" is that position."}</Prose>

<NeuralTable caption={"What exactly do we keep?"} headers={[<>{"Position"}</>,<>{"0"}</>,<>{"1"}</>,<>{"2"}</>,<>{"3"}</>,<>{"4"}</>,<>{"5"}</>,<>{"6"}</>,<>{"7"}</>,<>{"8"}</>]} rows={[[<>{"Token"}</>,<>{""}<code>{"<past>"}</code>{""}</>,<>{""}<code>{"l"}</code>{""}</>,<>{""}<code>{"a"}</code>{""}</>,<>{""}<code>{"c"}</code>{""}</>,<>{""}<code>{"t"}</code>{""}</>,<>{""}<code>{"a"}</code>{""}</>,<>{""}<code>{"t"}</code>{""}</>,<>{""}<code>{"e"}</code>{""}</>,<>{""}<code>{"<eos>"}</code>{""}</>],[<>{"Memory vector"}</>,<>{""}<InlineMath>{"h_0"}</InlineMath>{""}</>,<>{""}<InlineMath>{"h_1"}</InlineMath>{""}</>,<>{""}<InlineMath>{"h_2"}</InlineMath>{""}</>,<>{""}<InlineMath>{"h_3"}</InlineMath>{""}</>,<>{""}<InlineMath>{"h_4"}</InlineMath>{""}</>,<>{""}<InlineMath>{"h_5"}</InlineMath>{""}</>,<>{""}<InlineMath>{"h_6"}</InlineMath>{""}</>,<>{""}<InlineMath>{"h_7"}</InlineMath>{""}</>,<>{""}<InlineMath>{"h_8"}</InlineMath>{""}</>]]} />

<Prose>{"The two "}<code>{"a"}</code>{" positions have different notes. A forward recurrent encoder has read a different prefix by the time it reaches the second one. A saved vector is therefore more than a character ID; it may carry information about earlier characters as well."}</Prose>

<AttentionMemoryShelf />

<Prose>{"We keep both paths: final-state initialization and repeated reads. Attention does not recover information the encoder never represented, and mixing memories can itself lose distinctions. Its benefit is access to several saved representations through shorter, selectable paths."}</Prose>

<Prose>{"A "}<strong>{"bidirectional"}</strong>{" encoder also reads backward, so each saved position can contain information from both sides. Bahdanau's original translation system did this; our small experiment uses a forward encoder to keep the preceding lesson's source representation dimensions. Bidirectionality and the rule for reading memory are separate design choices. "}<a href={"https://arxiv.org/pdf/1409.0473"}>{"Original model, sections 3 and appendix A"}</a>{""}</Prose>

<H2>{"2. Build one read before building the whole network"}</H2>

<Prose>{"First separate finding information from returning it. A weather log might store a time beside each temperature. To ask for the temperature at 14:00, compare your requested time with the saved times, then return the associated temperature. The thing you compare is not the thing you return."}</Prose>

<AttentionLookupBridge />

<Prose>{"This gives us three names with distinct jobs:"}</Prose>

<ul><li>{"The "}<strong>{"query"}</strong>{" is what we are asking for: the requested time in that lookup, or the decoder's current need in our network."}</li><li>{"A "}<strong>{"key"}</strong>{" is what we compare with the query: a stored time, or a learned description of a source position."}</li><li>{"A "}<strong>{"value"}</strong>{" is what we read back: a temperature, or that position's feature vector."}</li></ul>

<Prose>{"An exact lookup gives one entry weight 1 and all others weight 0. An ordinary average gives every entry the same share. Attention allows the query to determine the shares. Softmax attention with finite scores gives every unmasked entry a positive share mathematically; floating-point underflow can round tiny shares to zero. “Pay attention” does not necessarily mean select just one entry."}</Prose>

<H3>{"Why read a mixture?"}</H3>

<Prose>{"If a winner-takes-all rule chooses A, a tiny score change that leaves A the winner changes nothing about the returned value. That discrete choice is hard to train through using ordinary derivatives. A smooth mixture can shift a little toward B as B's score rises. Output loss can then tell the scoring system whether that shift helped. Section 5 follows that learning signal. Hard selection and sparse reads are possible alternatives, but they need their own mechanisms."}</Prose>

<Prose>{"Our network uses numerical vectors rather than literal times. The decoder state supplies a query "}<InlineMath>{"q"}</InlineMath>{". Encoder features supply keys "}<InlineMath>{"k_j"}</InlineMath>{" for comparison and values "}<InlineMath>{"v_j"}</InlineMath>{" for the returned information. These are "}<strong>{"roles"}</strong>{"; basic recurrent attention often uses the same encoder vector as a value and as the input to a key projection. There need not be separate key and value storage."}</Prose>

<H3>{"Score: how well does each key match this query?"}</H3>

<Prose>{"Use small constructed vectors so every operation is visible. The two coordinates here have no claimed linguistic meaning. Memory A, B and C are labels for three rows, not words learned by the fitted model."}</Prose>

<NeuralTable caption={"Score: how well does each key match this query?"} headers={[<>{"Memory"}</>,<>{"Key "}<InlineMath>{"k_j"}</InlineMath>{""}</>,<>{"Value "}<InlineMath>{"v_j"}</InlineMath>{""}</>,<>{"Score "}<InlineMath>{"q^\\top k_j"}</InlineMath>{", for "}<InlineMath>{"q=(1,0)"}</InlineMath>{""}</>]} rows={[[<>{"A"}</>,<>{"(1,0)"}</>,<>{"(2,0)"}</>,<>{"1"}</>],[<>{"B"}</>,<>{"(0,1)"}</>,<>{"(0,2)"}</>,<>{"0"}</>],[<>{"C"}</>,<>{"(−1,0)"}</>,<>{"(−1,1)"}</>,<>{"−1"}</>]]} />

<Prose>{"The dot product multiplies corresponding coordinates and adds them. Query "}<InlineMath>{"(1,0)"}</InlineMath>{" reads the first coordinate of each key: A points with it, B contributes zero, and C points against it. For B, "}<InlineMath>{"1\\cdot0+0\\cdot1=0"}</InlineMath>{". A larger score will get a larger share. Scores can be negative; they are not probabilities. Dot products depend on vector length as well as direction."}</Prose>

<H3>{"Normalize: turn three scores into shares of one read"}</H3>

<Prose>{"We want nonnegative shares that add to 1. Exponentiating turns each score into a positive number; dividing each by their total gives those shares. For numerical stability, first subtract the largest score. The scores "}<InlineMath>{"(1,0,-1)"}</InlineMath>{" become "}<InlineMath>{"(0,-1,-2)"}</InlineMath>{", their exponentials are approximately "}<InlineMath>{"(1,0.367879,0.135335)"}</InlineMath>{", and the total is 1.503215."}</Prose>

<AttentionSoftmaxSteps />

<Prose>{"A's share is "}<InlineMath>{"1/1.503215"}</InlineMath>{", about 0.665241. The complete operation is "}<strong>{"softmax"}</strong>{":"}</Prose>

<div className="neural-equation"><MathBlock>{"\\alpha_j=\\frac{\\exp(e_j-\\max_i e_i)}{\\sum_i\\exp(e_i-\\max_i e_i)},\\qquad\n\\alpha=(0.665241,\\ 0.244728,\\ 0.090031)."}</MathBlock></div>

<Prose>{"Here "}<InlineMath>{"e_j"}</InlineMath>{" is a score, "}<InlineMath>{"\\alpha_j"}</InlineMath>{" its share, and "}<InlineMath>{"i"}</InlineMath>{" runs over every allowed memory. Subtracting the same maximum cancels between numerator and denominator, so it leaves the mathematical answer unchanged. Raising B's score changes A and C's shares too: they compete for the same total."}</Prose>

<H3>{"Read: multiply each value by its share, then add"}</H3>

<Prose>{"For the first coordinate, A contributes "}<InlineMath>{"0.665241\\times2"}</InlineMath>{", B contributes zero, and C contributes "}<InlineMath>{"0.090031\\times(-1)"}</InlineMath>{". For the second coordinate, B and C contribute. The same three shares apply to both coordinates:"}</Prose>

<div className="neural-equation"><MathBlock>{"c=\\sum_j\\alpha_jv_j\n=0.665241(2,0)+0.244728(0,2)+0.090031(-1,1)\n=(1.240451,\\ 0.579488)."}</MathBlock></div>

<Prose>{"The returned vector "}<InlineMath>{"c"}</InlineMath>{" is the "}<strong>{"context"}</strong>{". On the value plane below, the diamond is pulled toward the values with greater shares; it lies inside their triangle because the shares are nonnegative and sum to 1. The adjacent contribution table is the same calculation coordinate by coordinate."}</Prose>

<AttentionWorkedRead />

<Prose>{"We have completed a read: "}<strong>{"query → scores against keys → shared normalization → mixture of values"}</strong>{". It returns information, not yet a character. An output layer will combine that information with the decoder state to choose a character in section 3."}</Prose>

<AttentionReadLab />

<H2>{"3. Bahdanau and Luong: distinguish the score from the schedule"}</H2>

<Prose>{"A read alone cannot write "}<code>{"walked"}</code>{". It must sit inside a loop: the writer forms a question, reads source information, combines it with what has already been written, and produces a distribution over the next character. The chosen character becomes an input to the following step. The source notes stay fixed while the writer's state changes."}</Prose>

<Prose>{"There are two decisions inside that loop. "}<strong>{"How do we compare a question with a memory?"}</strong>{" That is the scorer. "}<strong>{"Do we read before or after updating the writer?"}</strong>{" That is the schedule. Bahdanau and Luong are historical names associated with particular choices; keeping these two questions separate makes the implementations easier to follow."}</Prose>

<H3>{"Additive scoring"}</H3>

<Prose>{"The query and the stored memory may use different feature systems. A decoder might have 96 coordinates while the encoder has 192. We cannot directly compare unequal lists with a dot product. A learned projection turns each into a shared comparison space, rather like giving two measurements a compatible set of features."}</Prose>

<AttentionAdditiveSteps />

<Prose>{"An additive score first projects the query and memory into a shared feature width "}<InlineMath>{"d_a"}</InlineMath>{", combines them, applies a nonlinearity, and reduces the result to one number:"}</Prose>

<div className="neural-equation"><MathBlock>{"e_{tj}=v_a^\\top\\tanh(W_q q_t+W_h h_j)."}</MathBlock></div>

<Prose>{"Here "}<InlineMath>{"W_q"}</InlineMath>{" has shape "}<InlineMath>{"d_a\\times d_s"}</InlineMath>{", "}<InlineMath>{"W_h"}</InlineMath>{" has shape "}<InlineMath>{"d_a\\times d_h"}</InlineMath>{", and "}<InlineMath>{"v_a"}</InlineMath>{" has "}<InlineMath>{"d_a"}</InlineMath>{" entries. Decoder width "}<InlineMath>{"d_s"}</InlineMath>{" and encoder width "}<InlineMath>{"d_h"}</InlineMath>{" can differ. We omit biases in this scoring layer; adding a bias inside the nonlinearity is another valid declared parameterization."}</Prose>

<Prose>{"Read the formula from the inside out. "}<InlineMath>{"W_q q_t"}</InlineMath>{" describes the current need; "}<InlineMath>{"W_h h_j"}</InlineMath>{" describes one saved position. Their sum is a comparison feature vector, not an attention weight. Applying "}<InlineMath>{"\\tanh"}</InlineMath>{" bends this combination so the effect of a memory can depend on the current query. Finally "}<InlineMath>{"v_a"}</InlineMath>{" combines the comparison features into one scalar score. Repeat the same scorer for every source position, then use the softmax and weighted read from section 2."}</Prose>

<Prose>{"The word “additive” refers to adding the projected query and memory before the nonlinearity. It does not mean adding the final attention probabilities."}</Prose>

<Prose>{"Writing "}<InlineMath>{"W[q;h]"}</InlineMath>{" inside the same "}<InlineMath>{"\\tanh"}</InlineMath>{" is equivalent: split the columns of "}<InlineMath>{"W"}</InlineMath>{" into "}<InlineMath>{"W_q"}</InlineMath>{" and "}<InlineMath>{"W_h"}</InlineMath>{". Concatenation itself does not make this version more expressive. The nonlinear concat score is explicitly shown in "}<a href={"https://arxiv.org/pdf/1508.04025v5"}>{"Luong et al.'s revised arXiv version, section 3.1"}</a>{"."}</Prose>

<H3>{"Dot and general scoring"}</H3>

<Prose>{"With equal query and memory widths, use"}</Prose>

<div className="neural-equation"><MathBlock>{"e_{tj}=q_t^\\top h_j\n\\quad\\text{(dot)}."}</MathBlock></div>

<Prose>{"To compare different widths, or learn a transformation before comparison, use"}</Prose>

<div className="neural-equation"><MathBlock>{"e_{tj}=q_t^\\top W_h h_j\n\\quad\\text{(general)}."}</MathBlock></div>

<Prose>{"For general scoring, "}<InlineMath>{"W_h"}</InlineMath>{" has shape "}<InlineMath>{"d_s\\times d_h"}</InlineMath>{". A layer mapping a 192-coordinate memory to a 96-coordinate query space is "}<strong>{"general attention"}</strong>{", even if its last operation is a dot product."}</Prose>

<Prose>{"The parameter comparison depends on dimensions. Without biases, "}<InlineMath>{"d_s=96,d_h=192,d_a=64"}</InlineMath>{" gives 18,496 parameters for additive scoring and 18,432 for general scoring. That is a difference of 64, not a factor of three. Dot scoring adds none when dimensions already match. This count excludes the encoder, decoder and output layers."}</Prose>

<H3>{"Two valid decoder timelines"}</H3>

<Prose>{"Suppose the already written prefix is "}<code>{"wal"}</code>{". A recurrent state already exists for the preceding step. We also know the most recent output character, "}<code>{"l"}</code>{". To write "}<code>{"k"}</code>{", we can either read using that existing state and then update it with "}<code>{"l"}</code>{" and the read, or first update with "}<code>{"l"}</code>{" and then use the new state to read. Both orders are executable. Asking for the new state before calculating the read, while also requiring that read to calculate the new state, would create a circular dependency."}</Prose>

<Prose>{"Let "}<InlineMath>{"s_{t-1}"}</InlineMath>{" be the previous decoder state and "}<InlineMath>{"E(y_{t-1})"}</InlineMath>{" the embedding of the previous output. At the first step, that output is a special start token, BOS."}</Prose>

<Prose>{""}<strong>{"Bahdanau-style order:"}</strong>{""}</Prose>

<ol start={1}><li>{"Query the source using "}<InlineMath>{"s_{t-1}"}</InlineMath>{"."}</li><li>{"Compute weights and context "}<InlineMath>{"c_t"}</InlineMath>{"."}</li><li>{"Update the recurrent state using the previous token and that context:"}</li></ol>

<Prose>{"   "}<InlineMath>{"s_t=\\operatorname{GRU}([E(y_{t-1});c_t],s_{t-1})"}</InlineMath>{"."}</Prose>

<ol start={4}><li>{"Use the new state and context to predict "}<InlineMath>{"y_t"}</InlineMath>{"."}</li></ol>

<Prose>{""}<strong>{"Luong-style order:"}</strong>{""}</Prose>

<ol start={1}><li>{"Update the recurrent state from the previous token:"}</li></ol>

<Prose>{"   "}<InlineMath>{"s_t=\\operatorname{GRU}(E(y_{t-1}),s_{t-1})"}</InlineMath>{"."}</Prose>

<ol start={2}><li>{"Query the source using this new state."}</li><li>{"Combine the state and returned context into an attentional vector:"}</li></ol>

<Prose>{"   "}<InlineMath>{"\\tilde s_t=\\tanh(W_c[s_t;c_t]+b_c)"}</InlineMath>{"."}</Prose>

<ol start={4}><li>{"Apply an output layer and softmax to predict "}<InlineMath>{"y_t"}</InlineMath>{"."}</li></ol>

<AttentionSchedules />

<Prose>{"Our teaching model uses the same form of attentional output projection in both orders. It uses native GRU cells, not the original papers' complete networks: the original Bahdanau decoder had a different output network, and Luong's experiments used stacked LSTMs. These simplifications are explicit so the experiment demonstrates mechanisms without impersonating a historical reproduction."}</Prose>

<H3>{"Input feeding remembers earlier reads"}</H3>

<Prose>{"A Luong decoder can feed "}<InlineMath>{"\\tilde s_{t-1}"}</InlineMath>{" alongside the previous token at the next step:"}</Prose>

<div className="neural-equation"><MathBlock>{"s_t=\\operatorname{GRU}([E(y_{t-1});\\tilde s_{t-1}],s_{t-1})."}</MathBlock></div>

<Prose>{"Initialize the fed vector to zero. This is an extra connection from the previous "}<strong>{"attentional vector"}</strong>{", not a replacement for the token embedding or a second pass through the source. It gives the state direct access to information from earlier attention decisions. It does not guarantee that each source position is covered once."}</Prose>

<Prose>{"The full program supports this connection as a separate extension. The reported fits keep it off for the general-scoring model; switching it on changes the decoder input width and requires a new fit."}</Prose>

<H3>{"Deeper scorer diagnosis: when the question cancels out"}</H3>

<Prose>{"This branch explains why the nonlinearity matters. You can continue to section 4 after understanding one complete decoder step, then return to this counterexample."}</Prose>

<Prose>{"Suppose someone tries to learn the score with a linear layer on the concatenated query and key:"}</Prose>

<div className="neural-equation"><MathBlock>{"e_j=a^\\top q+b^\\top k_j."}</MathBlock></div>

<Prose>{"The first term is the same for every memory. Softmax cancels it:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{\\exp(a^\\top q+b^\\top k_j)}\n{\\sum_i\\exp(a^\\top q+b^\\top k_i)}\n=\\frac{\\exp(b^\\top k_j)}{\\sum_i\\exp(b^\\top k_i)}."}</MathBlock></div>

<Prose>{"The weights no longer depend on the question. The code can run, gradients can exist elsewhere, and the model can appear to “have attention,” yet this scoring rule cannot change its read according to the query. A nonlinear interaction or a query–key product repairs this particular limitation. This is why the details inside a small scoring formula matter."}</Prose>

<AttentionCancellationLab />

<H2>{"4. Put the read into tensors and code"}</H2>

<Prose>{"We have followed one example and one output step. Batching just lays several examples side by side in rectangular arrays; it does not change the meaning of query, memory or read. The bookkeeping matters because an array may contain storage cells that are not real source positions."}</Prose>

<Prose>{"Suppose a batch has "}<InlineMath>{"B"}</InlineMath>{" examples, padded source width "}<InlineMath>{"S"}</InlineMath>{", padded target width "}<InlineMath>{"T"}</InlineMath>{", memory width "}<InlineMath>{"d_h"}</InlineMath>{", and decoder width "}<InlineMath>{"d_s"}</InlineMath>{"."}</Prose>

<NeuralTable caption={"4. Put the read into tensors and code"} headers={[<>{"Object"}</>,<>{"Shape"}</>,<>{"Meaning"}</>]} rows={[[<>{"Source IDs"}</>,<>{""}<InlineMath>{"B\\times S"}</InlineMath>{""}</>,<>{"Embedding addresses, including request and source EOS"}</>],[<>{"Source lengths"}</>,<>{""}<InlineMath>{"B"}</InlineMath>{""}</>,<>{"Number of actual tokens in each source"}</>],[<>{"Encoder memory"}</>,<>{""}<InlineMath>{"B\\times S\\times d_h"}</InlineMath>{""}</>,<>{"One feature vector per stored source position"}</>],[<>{"One query"}</>,<>{""}<InlineMath>{"B\\times d_s"}</InlineMath>{""}</>,<>{"One current question per example"}</>],[<>{"One attention row"}</>,<>{""}<InlineMath>{"B\\times S"}</InlineMath>{""}</>,<>{"Distribution over valid source positions"}</>],[<>{"One context"}</>,<>{""}<InlineMath>{"B\\times d_h"}</InlineMath>{""}</>,<>{"Weighted memory read"}</>],[<>{"All attention rows"}</>,<>{""}<InlineMath>{"B\\times T\\times S"}</InlineMath>{""}</>,<>{"One source distribution per decoder step"}</>],[<>{"Output logits"}</>,<>{""}<InlineMath>{"B\\times T\\times V"}</InlineMath>{""}</>,<>{"Scores over output vocabulary, not source positions"}</>]]} />

<Prose>{"Attention and output softmax normalize over different things. A model can put 99% attention on one source position while being uncertain among several output characters."}</Prose>

<H3>{"Source padding, target padding and output constraints do different jobs"}</H3>

<Prose>{"A short source is padded to share a rectangular batch with longer sources. PAD is storage, not another observation. Set invalid source scores to "}<InlineMath>{"-\\infty"}</InlineMath>{" "}<strong>{"before"}</strong>{" softmax. The remaining valid positions receive the full probability mass."}</Prose>

<Prose>{"Imagine two real memories with equal scores and scalar values 2 and 4. Their read is "}<InlineMath>{"(2+4)/2=3"}</InlineMath>{". Add a padding slot with score 0 and value 0, but forget the mask: all three scores are still equal, so the read becomes "}<InlineMath>{"(2+4+0)/3=2"}</InlineMath>{". Nothing about the real observations changed. Their shares shrank because storage entered the competition."}</Prose>

<AttentionPaddingShares />

<Prose>{"Setting padded values to zero while leaving their scores valid is insufficient. A zero-valued memory can still steal probability mass and shrink the context. In a bidirectional encoder, processing padding can also change actual backward states; an attention mask cannot undo that earlier contamination. Our program packs actual source lengths before running the encoder, then also masks the attention read."}</Prose>

<Prose>{"Each source here includes a request token and EOS, so there is always a valid memory position. A general-purpose reader must reject or explicitly handle an all-masked row: ordinary softmax on all "}<InlineMath>{"-\\infty"}</InlineMath>{" values is undefined."}</Prose>

<Prose>{"Target padding has a separate role. For the reference "}<code>{"cared<EOS>"}</code>{", teacher-forced decoder inputs are "}<code>{"<BOS>cared"}</code>{". Cross-entropy scores each next target, ignores padded target cells, and averages over the remaining target tokens. Source EOS is a memory; target EOS is a predicted stopping decision. BOS and request tokens are not valid generated outputs in this task."}</Prose>

<AttentionFittedLab mode="padding" />

<Prose>{"A source edit invalidates its cache. A decoder-prefix edit can reuse source memory but must recompute the affected decoder suffix. Renaming a display label changes neither."}</Prose>

<H3>{"Implement exactly that read"}</H3>

<Prose>{"This small NumPy program implements one dot-product read. "}<code>{"keys"}</code>{" has one row per source position; "}<code>{"values"}</code>{" may have a different feature width. The validity mask says which rows contain actual source data. Finite arrays with compatible dimensions are the function's input contract. The explicit all-masked check protects the one undefined case in the normalization."}</Prose>

<CodeBlock language={"python"}>{"import numpy as np\n\ndef attention_read(query, keys, values, valid):\n    if not np.any(valid):\n        raise ValueError(\"At least one source position must be valid\")\n    scores = keys @ query\n    scores = np.where(valid, scores, -np.inf)\n    masses = np.exp(scores - np.max(scores))\n    weights = masses / masses.sum()\n    return weights, weights @ values\n\nkeys = np.array([[1., 0.], [0., 1.], [-1., 0.]])\nvalues = np.array([[2., 0.], [0., 2.], [-1., 1.]])\nweights, context = attention_read(\n    np.array([1., 0.]), keys, values, np.array([True, True, True])\n)\nprint(np.round(weights, 6), np.round(context, 6))"}</CodeBlock>

<Prose>{"The output is "}<code>{"[0.665241 0.244728 0.090031] [1.240451 0.579488]"}</code>{", matching section 2. "}<code>{"keys @ query"}</code>{" computes all scores; "}<code>{"weights @ values"}</code>{" computes all context coordinates. Everything between them normalizes only allowed memories. The implementation uses vectorized matrix operations and stores one score row; it does not construct a source-by-source matrix for this one-query operation."}</Prose>

<Prose>{"The library route uses the same operations on tensors: a matrix product for scores, "}<code>{"masked_fill"}</code>{" before "}<code>{".softmax"}</code>{", and a weighted matrix product for the read. PyTorch then differentiates those operations as part of a custom "}<code>{"nn.Module"}</code>{". Section 6 supplies the complete trainable model; section 7 links the full NumPy reconstruction. A library multi-head layer changes the architecture, so simply importing it would not reproduce this recurrent decoder."}</Prose>

<H2>{"5. Learn which read helps the output"}</H2>

<Prose>{"Training does not ordinarily come with labels saying which input position to look at. It comes with desired outputs. If a different read would lower the output loss, gradients adjust the encoder, scoring parameters and decoder."}</Prose>

<Prose>{"Think of each source value as proposing information to add to the current mixture. If giving B a little more share would make the desired output more likely, training should raise B's relative score. If the same change would hurt, it should lower it. The feedback depends on what B contains, not just on whether B already has a large weight."}</Prose>

<Prose>{"Return to our three-memory arithmetic. Pretend the two context coordinates are logits for two classes, and the correct class is the second one. This tiny output head makes the whole path visible:"}</Prose>

<div className="neural-equation"><MathBlock>{"p=\\operatorname{softmax}(c)=(0.659477,\\ 0.340523),\\qquad\nL=-\\log p_2=1.077272."}</MathBlock></div>

<Prose>{"The first coordinate currently wins, although we want the second class. A's value "}<InlineMath>{"(2,0)"}</InlineMath>{" favors the first class, so putting more weight there would hurt. B's "}<InlineMath>{"(0,2)"}</InlineMath>{" favors the second; C's "}<InlineMath>{"(-1,1)"}</InlineMath>{" also favors it. Moving share away from A toward B or C therefore moves the output in a useful direction. That is the intuition the derivative below will make precise."}</Prose>

<AttentionLearningDirection />

<H3>{"Deeper calculation: follow one gradient all the way back"}</H3>

<Prose>{"The sign argument above is enough to follow the training story on a first pass. For implementing or checking the backward pass, this derivation supplies the exact quantities used by the next lab."}</Prose>

<Prose>{"For softmax cross-entropy, the derivative with respect to these logits is"}</Prose>

<div className="neural-equation"><MathBlock>{"g=\\frac{\\partial L}{\\partial c}=p-(0,1)\n=(0.659477,-0.659477)."}</MathBlock></div>

<Prose>{"We want less first-coordinate support and more second-coordinate support. How should a source score change?"}</Prose>

<Prose>{"Differentiate the softmax-weighted sum:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{\\partial c}{\\partial e_j}=\\alpha_j(v_j-c),\n\\qquad\n\\frac{\\partial L}{\\partial e_j}\n=\\alpha_j\\,g^\\top(v_j-c)."}</MathBlock></div>

<Prose>{"The derivative compares each value with the "}<strong>{"current mixture"}</strong>{", not just its attention weight. Here the three score gradients are"}</Prose>

<div className="neural-equation"><MathBlock>{"(0.587450,\\ -0.429460,\\ -0.157990)."}</MathBlock></div>

<Prose>{"Gradient descent therefore reduces A's score and raises B's and C's. With dot scoring "}<InlineMath>{"e_j=q^\\top k_j"}</InlineMath>{","}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{\\partial L}{\\partial q}\n=\\sum_j\\frac{\\partial L}{\\partial e_j}k_j\n=(0.745440,-0.429460)."}</MathBlock></div>

<Prose>{"One learning-rate 0.1 step changes "}<InlineMath>{"q=(1,0)"}</InlineMath>{" to approximately "}<InlineMath>{"(0.925456,0.042946)"}</InlineMath>{". Recomputing the entire read gives loss 1.003220, down from 1.077272. These are calculated values, independently matched to automatic differentiation and finite differences. A sufficiently large step need not lower the loss."}</Prose>

<AttentionGradientFlow />

<AttentionReadLab learning />

<Prose>{"Attention gives a direct weighted path from an output's loss to stored source values. Recurrent dependencies still exist; this additional path does not make vanishing gradients, optimization difficulties or generalization errors impossible."}</Prose>

<H2>{"6. Does revisiting memory help on real spellings?"}</H2>

<Prose>{"Return to the opening problem: can a writer generate an unfamiliar spelling more reliably if it can consult the source again? We can now test that with trained models, rather than infer success from an attractive attention picture. The unit of success is a complete requested word, including a correct stopping decision."}</Prose>

<Prose>{"Use the same small English inflection extract as the preceding lesson: 1,800 records from 600 selected spellings, with three requested verb forms each. The source is the "}<a href={"https://github.com/unimorph/eng/tree/66e0e9e8e2dcd196da081a25a48e5c1fe3d8b49b"}>{"pinned UniMorph English repository"}</a>{". Its README names Wikipedia and licenses the data under "}<a href={"https://creativecommons.org/licenses/by-sa/3.0/"}>{"CC BY-SA 3.0"}</a>{". The supplied extract retains attribution, source rows, filtering decisions and license in "}<a href={"/learn-code/attention-mechanism-bahdanau-luong/data-provenance.md"}>{"data provenance"}</a>{"."}</Prose>

<Prose>{"All requests for the same lemma stay together. Selected lemmas sharing a target form are also grouped together, preventing the demonstrated "}<code>{"work"}</code>{"/"}<code>{"worke"}</code>{" overlap. The final partition has 451 training lemmas/1,353 rows and 149 development lemmas/447 rows, with no shared lemma spelling or target form across those partitions. This is a conservative identity check, not a proof that the lexicon contains no other aliases."}</Prose>

<Prose>{"The development data has already been inspected in the earlier lesson and is inspected here. It remains development data. These experiments are useful for comparing mechanisms and diagnosing errors; they do not supply an untouched final test."}</Prose>

<H3>{"Keep the task fixed and declare the architecture changes"}</H3>

<Prose>{"Inputs are a request token, lowercase letters and source EOS. Targets are the recorded form and target EOS. The 32-token vocabulary is declared from the alphabet and special tokens. Each model uses 24-coordinate embeddings and a 64-coordinate forward GRU encoder. No pretrained downloads or GPU are needed."}</Prose>

<Prose>{"The preceding fixed-context model initializes a 64-coordinate decoder from the final encoder state. The new models preserve that initialization and add attention:"}</Prose>

<NeuralTable caption={"Keep the task fixed and declare the architecture changes"} headers={[<>{"Model"}</>,<>{"Query/order"}</>,<>{"Other changes"}</>,<>{"Parameters"}</>]} rows={[[<>{"Fixed context"}</>,<>{"No repeated source read"}</>,<>{"Previous lesson's decoder and linear output head"}</>,<>{"37,408"}</>],[<>{"Additive attention"}</>,<>{"Previous state; read before update"}</>,<>{"32-coordinate scoring layer; context enters GRU; combined output head"}</>,<>{"62,080"}</>],[<>{"General attention"}</>,<>{"New state; read after update"}</>,<>{"Learned 64→64 key map; combined output head; no input feeding"}</>,<>{"49,760"}</>]]} />

<Prose>{"The comparison changes complete declared architectures, including parameter counts and output heads. It is not an isolated proof that a score function alone causes every difference. Matched dimensions, data, sampling schedule and training budget make it useful; they do not eliminate every confound."}</Prose>

<Prose>{"All neural runs use seeds 1, 2 and 3, Adam learning rate 0.003, 1,200 updates, 64 examples sampled with replacement per update and global gradient clipping at norm 1. The sampling generator is seeded with 100 plus the run seed. The same seed does not imply equal initial parameters across differently shaped networks. Evaluate fixed checkpoints without choosing an early stopping point after seeing them. Decode greedily, allowing at most 16 generated tokens including EOS."}</Prose>

<Prose>{"Retain the simple suffix rules from the previous lesson. They handle common "}<code>{"e"}</code>{" and consonant-plus-"}<code>{"y"}</code>{" endings but have no irregular-word lookup or general consonant-doubling rule. A useful model must compete with that task knowledge, not only another neural network."}</Prose>

<H3>{"Actual results"}</H3>

<Prose>{"“Exact” requires the recorded form and natural EOS. Character error rate is total insertions, deletions and substitutions divided by total reference characters, excluding EOS. The 447 development references contain 3,490 characters."}</Prose>

<Prose>{"Read one comparison first. After the same number of updates, fixed-context seed 1 produces 53 of 447 development forms exactly; additive-attention seed 1 produces 395. Simple suffix rules produce 407. In this experiment, a repeated source read helps the neural model substantially, but does not beat the task-specific rules. The table retains all seeds so one selected run does not stand in for the whole result."}</Prose>

<NeuralTable caption={"Actual results"} headers={[<>{"Model / seed"}</>,<>{"Training exact /1,353"}</>,<>{"Development exact /447"}</>,<>{"Development character edits"}</>,<>{"Character error rate"}</>]} rows={[[<>{"Predeclared suffix rules"}</>,<>{"1,206"}</>,<>{"407"}</>,<>{"57"}</>,<>{"0.01633"}</>],[<>{"Fixed context /1"}</>,<>{"1,328"}</>,<>{"53"}</>,<>{"1,517"}</>,<>{"0.43467"}</>],[<>{"Fixed context /2"}</>,<>{"1,323"}</>,<>{"41"}</>,<>{"1,533"}</>,<>{"0.43926"}</>],[<>{"Fixed context /3"}</>,<>{"1,287"}</>,<>{"52"}</>,<>{"1,475"}</>,<>{"0.42264"}</>],[<>{"Additive /1"}</>,<>{"1,338"}</>,<>{"395"}</>,<>{"86"}</>,<>{"0.02464"}</>],[<>{"Additive /2"}</>,<>{"1,327"}</>,<>{"380"}</>,<>{"108"}</>,<>{"0.03095"}</>],[<>{"Additive /3"}</>,<>{"1,352"}</>,<>{"387"}</>,<>{"97"}</>,<>{"0.02779"}</>],[<>{"General /1"}</>,<>{"1,292"}</>,<>{"345"}</>,<>{"165"}</>,<>{"0.04728"}</>],[<>{"General /2"}</>,<>{"1,353"}</>,<>{"392"}</>,<>{"90"}</>,<>{"0.02579"}</>],[<>{"General /3"}</>,<>{"1,348"}</>,<>{"385"}</>,<>{"101"}</>,<>{"0.02894"}</>]]} />

<Prose>{"The attentive models generalize to many more unseen spellings than the fixed-context runs under this protocol. The rule baseline still has the highest exact-match count and the fewest character edits. Treat that as useful evidence about this task and data budget. It is not an inconvenience to tune away."}</Prose>

<Prose>{"The outcomes also separate fitting from transfer. General seed 2 gets every training form right but misses 55 development forms. Additive seed 3 fits 1,352 training records but does not produce the best development result. Reading the source again helps, while memorizing training outputs remains possible."}</Prose>

<AttentionLearningCurves />

<Prose>{"Length is another diagnostic, not a universal capacity threshold. Additive seed 1 gets 145/168 shorter lemmas and 250/279 longer lemmas correct. These groups differ in spelling patterns and examples, not just length. An increase in longer-word accuracy does not mean length is intrinsically easier, just as a decrease would not by itself prove a fixed memory limit."}</Prose>

<H3>{"Run the saved model first"}</H3>

<Prose>{"Download "}<a href={"/learn-code/attention-mechanism-bahdanau-luong/attentive-inflection.py"}>{"attentive-inflection.py"}</a>{", "}<a href={"/learn-code/attention-mechanism-bahdanau-luong/calculated-inputs.json"}>{"calculated-inputs.json"}</a>{" and "}<a href={"/learn-code/attention-mechanism-bahdanau-luong/english-inflections.csv"}>{"english-inflections.csv"}</a>{" into one directory. The JSON contains actual trained parameters and recorded results; it is an offline input, not a request to train during page rendering."}</Prose>

<Prose>{"Use a Python environment with NumPy and CPU PyTorch. The author run used Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu. Compatible versions can run the program, but numerical kernels and training trajectories can differ. From that directory, save and run this complete small example:"}</Prose>

<CodeBlock language={"python"}>{"from pathlib import Path\nimport importlib.util\nimport json\nimport sys\nsys.dont_write_bytecode = True\nimport torch\n\nroot = Path.cwd()\nspec = importlib.util.spec_from_file_location(\"inflection\", root/\"attentive-inflection.py\")\ninflection = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(inflection)\ntorch.set_num_threads(1)\nreport = json.loads((root/\"calculated-inputs.json\").read_text(encoding=\"utf-8\"))\nrun = next(item for item in report[\"runs\"] if item[\"kind\"] == \"additive\" and item[\"seed\"] == 1)\nmodel = inflection.AttentiveInflector(\"additive\")\nshapes = model.state_dict()\nmodel.load_state_dict({\n    name: torch.tensor(value, dtype=shapes[name].dtype)\n    for name, value in run[\"weights\"].items()\n})\nresult = inflection.greedy(model, [{\"lemma\": \"lactate\", \"feature\": \"past\"}])[0]\nprint(result[\"prediction\"], result[\"ended_with_eos\"])"}</CodeBlock>

<Prose>{"The executed result is "}<code>{"lactated True"}</code>{". Generation receives the lemma and request; it never receives "}<code>{"lactated"}</code>{" as a reference input. Change the request or spelling and inspect how the generated answer changes. A constructed spelling has no automatic correctness label just because the model returns something."}</Prose>

<H3>{"Read and optionally run the complete training program"}</H3>

<Prose>{"The full program below creates all inputs, masks, model components, optimizer steps, evaluations and saved parameters. To reproduce training, save it as "}<code>{"attentive-inflection.py"}</code>{" beside the CSV and run "}<code>{"python attentive-inflection.py"}</code>{". It runs six fits and writes "}<code>{"calculated-inputs.json"}</code>{"; preserve the supplied file under another name if you want to compare your run with it."}</Prose>

<Prose>{"Read "}<code>{"encode"}</code>{" first: it packs real source lengths and returns memory, precomputed keys, a validity mask and an initial state. Then read "}<code>{"step"}</code>{": the two branches put the attention read on different sides of the recurrent update. "}<code>{"forward"}</code>{" feeds reference prefixes for likelihood training; "}<code>{"greedy"}</code>{" feeds generated prefixes. "}<code>{"assess"}</code>{" reports both, so low teacher-forced loss is not silently equated with correct free generation."}</Prose>

<AttentionProgram file="attentive-inflection.py" title="Read the complete CPU training and generation program" />

<H2>{"7. Connect a real read to a generated character"}</H2>

<Prose>{"The opening diagram showed a behavior we wanted: a changing need should lead to a changing read. Now we can inspect actual trained weights. In an alignment picture, each row answers “where did this output step read?” and each column identifies one saved source position. Read across one row before comparing rows; a bright cell is a large source share, not the probability of a correct output."}</Prose>

<Prose>{"The seed 1 additive model produces "}<code>{"lactated<EOS>"}</code>{" for the worked input. Its actual attention rows put the greatest weight on these source positions:"}</Prose>

<NeuralTable caption={"7. Connect a real read to a generated character"} headers={[<>{"Output being predicted"}</>,<>{"Highest-weight source position"}</>,<>{"That weight"}</>,<>{"Probability of emitted output"}</>]} rows={[[<>{""}<code>{"l"}</code>{""}</>,<>{"1: "}<code>{"l"}</code>{""}</>,<>{"0.9755"}</>,<>{"0.9998"}</>],[<>{""}<code>{"a"}</code>{""}</>,<>{"2: first "}<code>{"a"}</code>{""}</>,<>{"0.8894"}</>,<>{"0.9994"}</>],[<>{""}<code>{"c"}</code>{""}</>,<>{"3: "}<code>{"c"}</code>{""}</>,<>{"0.9056"}</>,<>{"0.9999"}</>],[<>{""}<code>{"t"}</code>{""}</>,<>{"4: first "}<code>{"t"}</code>{""}</>,<>{"0.9047"}</>,<>{"0.9999"}</>],[<>{""}<code>{"a"}</code>{""}</>,<>{"5: second "}<code>{"a"}</code>{""}</>,<>{"0.6200"}</>,<>{"0.9804"}</>],[<>{""}<code>{"t"}</code>{""}</>,<>{"6: second "}<code>{"t"}</code>{""}</>,<>{"0.8527"}</>,<>{"0.9999"}</>],[<>{""}<code>{"e"}</code>{""}</>,<>{"7: "}<code>{"e"}</code>{""}</>,<>{"0.4830"}</>,<>{"1.0000, rounded"}</>],[<>{""}<code>{"d"}</code>{""}</>,<>{"0: "}<code>{"<past>"}</code>{""}</>,<>{"0.7016"}</>,<>{"0.9998"}</>],[<>{""}<code>{"<eos>"}</code>{""}</>,<>{"6: second "}<code>{"t"}</code>{""}</>,<>{"0.3358"}</>,<>{"1.0000, rounded"}</>]]} />

<AttentionWorkedAlignment />

<Prose>{"The picture is consistent with copying much of the spelling and consulting the request while appending "}<code>{"d"}</code>{". The EOS row is more diffuse and does not peak at source EOS in this model. That does not make its prediction invalid: contextual features and decoder state can supply stopping information in other ways."}</Prose>

<Prose>{"Do not turn that plausible reading into a claim of uniquely discovered linguistic rules. A memory for one position can encode information from other positions. The output also depends on decoder state and previous outputs, and the value vectors matter in addition to their weights."}</Prose>

<Prose>{"Here is an exact counterexample to “the attention distribution uniquely explains the answer”:"}</Prose>

<div className="neural-equation"><MathBlock>{"v_1=(1,0),\\quad v_2=(0,1),\\quad v_3=(0.5,0.5)."}</MathBlock></div>

<Prose>{"Both "}<InlineMath>{"\\alpha=(0.4,0.4,0.2)"}</InlineMath>{" and "}<InlineMath>{"\\alpha'=(0.2,0.2,0.6)"}</InlineMath>{" produce "}<InlineMath>{"c=(0.5,0.5)"}</InlineMath>{". Their heatmaps look different, but a downstream calculation receiving only this context and the same other inputs cannot distinguish them. These are valid softmax outcomes: scores equal to their log probabilities would produce them."}</Prose>

<AttentionAmbiguityFigure />

<AttentionFittedLab />

<Prose>{"The masking investigation also has an instructive result: admitting extra zero memories can alter attention and probabilities while leaving the greedy word unchanged. A correct-looking word is not enough to prove the tensor computation is correct. Conversely, two different weights need not imply two different words. Inspect the quantity that your hypothesis actually concerns."}</Prose>

<H2>{"Implement the read, then compose it into ordinary training"}</H2>

<Prose>{"The complete "}<a href={"/learn-code/attention-mechanism-bahdanau-luong/attentive-inflection.py"}>{"attentive-inflection.py"}</a>{" exposes additive and general scoring, source masks, normalized weighted reads and the decoder schedule. "}<a href={"/learn-code/attention-mechanism-bahdanau-luong/attention-calculations.py"}>{"attention-calculations.py"}</a>{" opens the small numerical derivatives and local-window/copy calculations, while "}<a href={"/learn-code/attention-mechanism-bahdanau-luong/attention-mechanics.py"}>{"attention-mechanics.py"}</a>{" reconstructs saved weights through NumPy and compares its trace with the native model. The scores, weights and context all remain learner-visible. The normal library route is this custom "}<code>{"nn.Module"}</code>{" composed with "}<code>{"nn.GRU"}</code>{", embeddings, losses and optimizers; "}<code>{"nn.MultiheadAttention"}</code>{" is a different architecture and should not be presented as a drop-in Bahdanau decoder."}</Prose>

<AttentionScratchRoute />

<Prose>{"The encoder's projected keys can be cached because the source and scorer parameters stay fixed during one decoding call. Cache them "}<strong>{"after"}</strong>{" the source mask has been paired with that source, and rebuild them after a parameter update or source edit. The decoder query changes every step. For source length S, target length T, memory width H, query width D and scoring width A, project the source once in O(SHA). Each output step costs O(DA + SA + SH) for query projection, scores and value mixing. Cached keys occupy O(SA), beside O(SH) source memory. These costs exclude the recurrent update and vocabulary projection. That is a useful avoided recomputation, not a claim to remove attention's source scan."}</Prose>

<Prose>{"For a research change, add a positive scoring temperature before masked softmax, not after the weighted sum. The same padding mask must still exclude invalid positions exactly; temperature does not legalize padding. With general scores "}<code>{"[0, log(3)]"}</code>{" and values "}<code>{"[2,10]"}</code>{", temperature1 gives weights[0.25,0.75] and context8; temperature2 gives weights proportional to[1,√3] and context approximately7.0718. Smaller temperature sharpens competition but can make gradients and alignments less forgiving; it is not an accuracy guarantee."}</Prose>

<details>

<summary>Implementation hint</summary>

<Prose>{"Divide scores by temperature, then apply the illegal-key mask before softmax. Reject a row with no legal source token."}</Prose>

</details>

<details>

<summary>Solution and success criteria</summary>

<Prose>{"Implement "}<code>{"weights = (scores / temperature).masked_fill(~valid, -inf).softmax(-1)"}</code>{" for strictly positive temperature. At temperature2, context is "}<code>{"(2 + 10*sqrt(3))/(1+sqrt(3)) ≈7.0718"}</code>{". Recompute one output and its score gradient using the same saved source/query parameters in both NumPy and Torch. Append a large-valued masked source token and confirm it changes neither context nor output. Do not compare two different random fits and call that temperature-implementation parity."}</Prose>

</details>

<Prose>{"The extended topics in the next section remain distinct: exact hard-monotonic alignment algorithms, a complete pointer-generator training system and location-sensitive speech models need their own objectives and datasets. The local copy-aggregation and normalized-window calculations are complete bounded mechanisms; they do not secretly claim those whole systems have been trained here."}</Prose>

<H2>{"8. Deeper branches: other reads and other tasks"}</H2>

<Prose>{"This section is optional on the first pass. Each branch changes one part of the source-read idea; it does not add a new requirement before beginning the next core topic."}</Prose>

<H3>{"Global, local and monotonic are different constraints"}</H3>

<Prose>{"If a recording contains thousands of frames but the next character depends mostly on a nearby region, rereading every frame may waste work. A window limits the search. The tradeoff is visible: an excluded memory cannot help, even when it contains the best match. This is a choice about which memories are available, separate from how the allowed ones are scored."}</Prose>

<Prose>{"Global attention considers every valid source position at every target step. For a source of length "}<InlineMath>{"S"}</InlineMath>{" and output of length "}<InlineMath>{"T"}</InlineMath>{", that creates "}<InlineMath>{"S T"}</InlineMath>{" score comparisons."}</Prose>

<Prose>{"Local attention restricts available positions to a window. A window centered at 3 with radius 2 on positions 1–5 contains all five; radius 1 contains only 2, 3 and 4. The decoder cannot consult a distant position outside that window even if it would have received the largest global score."}</Prose>

<Prose>{"Luong's local-p predicts a real center from the current state and multiplies its windowed alignment by a Gaussian factor. In the paper's one-based position convention,"}</Prose>

<div className="neural-equation"><MathBlock>{"p_t=S\\,\\sigma(v_p^\\top\\tanh(W_p s_t)),\\qquad\nw_{tj}=\\alpha_{tj}\\exp\\left[-\\frac{(j-p_t)^2}{2(D/2)^2}\\right]."}</MathBlock></div>

<Prose>{"The window clips at source boundaries. Membership changes when the center crosses a boundary; the construction is differentiable almost everywhere, not everywhere. The original equation multiplies by the Gaussian "}<strong>{"without an additional normalization in that displayed formula"}</strong>{". Those resulting weights need not sum to one. "}<a href={"https://arxiv.org/pdf/1508.04025v5"}>{"Revised section 3.2"}</a>{""}</Prose>

<Prose>{"For a constructed example, positions 1–5 have scores "}<InlineMath>{"(0,0.5,1,-0.5,2)"}</InlineMath>{" and scalar values equal to their position numbers. Center 3, radius 2 gives post-Gaussian weights approximately"}</Prose>

<div className="neural-equation"><MathBlock>{"(0.010128,0.074836,0.203425,0.027531,0.074836),"}</MathBlock></div>

<Prose>{"with sum 0.390755 and context 1.254375. Explicitly renormalizing those weights changes the context to 3.210133. Renormalization is a valid alternative design, but it changes the read's magnitude. Label the formula in use."}</Prose>

<AttentionWindowLab />

<Prose>{"A local window can move backward; locality alone does not enforce monotonicity. A monotonic mechanism constrains movement through source order. That can suit speech, while unrestricted reordering is useful in translation. Availability is another condition: a bidirectional encoder or a read of the entire future source cannot become streaming merely because its display uses a narrow window."}</Prose>

<H3>{"Copy a name that the output vocabulary does not contain"}</H3>

<Prose>{"An ordinary output softmax can emit only vocabulary entries. Looking closely at an unfamiliar source name does not automatically create a new output token."}</Prose>

<Prose>{"A pointer-generator adds a copying route. Let "}<InlineMath>{"p_{\\mathrm{gen}}"}</InlineMath>{" be the learned probability of using the vocabulary route. Sum attention over "}<strong>{"all occurrences"}</strong>{" of a word to obtain its copy mass, then mix:"}</Prose>

<div className="neural-equation"><MathBlock>{"P(w)=p_{\\mathrm{gen}}P_{\\mathrm{vocab}}(w)\n+(1-p_{\\mathrm{gen}})\\sum_{j:x_j=w}\\alpha_j."}</MathBlock></div>

<Prose>{"For source "}<code>{"Ada met Ada"}</code>{", attention "}<InlineMath>{"(0.2,0.3,0.5)"}</InlineMath>{", vocabulary probabilities "}<code>{"Ada:0.1, met:0.6, left:0.3"}</code>{", and "}<InlineMath>{"p_{\\mathrm{gen}}=0.4"}</InlineMath>{", the final probabilities are "}<code>{"Ada:0.46, met:0.42, left:0.12"}</code>{". Ada's two positions contribute 0.7 copy mass. If Ada were outside the vocabulary, its vocabulary contribution would be zero while its copy route could remain available."}</Prose>

<AttentionCopyFlow />

<H3>{"Remember where a speech reader has been"}</H3>

<Prose>{"Imagine transcribing a repeated syllable. Two regions of the recording can have very similar acoustic content. Asking only “does this sound like what I need?” may not distinguish the earlier occurrence from the next occurrence. Remembering where the last read was provides another clue."}</Prose>

<Prose>{"Speech contains repeated and similar acoustic fragments. A content score can be ambiguous when two memories look similar. A location-aware scorer adds features from the previous attention row:"}</Prose>

<div className="neural-equation"><MathBlock>{"f_t=F*\\alpha_{t-1},\\qquad\ne_{tj}=v^\\top\\tanh(W_s s_{t-1}+W_hh_j+W_f f_{tj}+b)."}</MathBlock></div>

<Prose>{"The convolution "}<InlineMath>{"F*\\alpha_{t-1}"}</InlineMath>{" measures local patterns around each position in the previous read. It lets a score depend on both current content and recent location. This differs from permanently labeling one position “already used.” "}<a href={"https://arxiv.org/pdf/1506.07503"}>{"Chorowski et al., section 2.2"}</a>{""}</Prose>

<AttentionLocationFlow />

<Prose>{"For example, two acoustic regions can resemble the same vowel. A previous attention peak near the first supplies positional evidence about which occurrence the decoder may be approaching. Learned weights decide how much to use that evidence; the formula alone does not prohibit jumps or guarantee alignment."}</Prose>

<Prose>{"Listen, Attend and Spell combines acoustic encoding with character generation. Its pyramidal bidirectional encoder reduces source resolution before reading it, so acoustic frame count need not match output character count. Its original full-input architecture is not inherently streaming. "}<a href={"https://arxiv.org/pdf/1508.01211"}>{"LAS, sections 3–3.1"}</a>{""}</Prose>

<Prose>{"A separate "}<strong>{"coverage"}</strong>{" vector "}<InlineMath>{"u_{tj}=\\sum_{\\tau<t}\\alpha_{\\tau j}"}</InlineMath>{" records accumulated reads. It can enter a score, or an overlap penalty "}<InlineMath>{"\\sum_j\\min(\\alpha_{tj},u_{tj})"}</InlineMath>{" can discourage repetition. Requiring exactly one unit per source is inappropriate for tasks such as summarization, where some material should be omitted and a phrase may support several output words. "}<a href={"https://arxiv.org/pdf/1704.04368"}>{"Coverage mechanism, section 2.3"}</a>{""}</Prose>

<Prose>{"Think of coverage as a running tally beside the memory shelf. Every output distributes one unit of attention across the source, so two completed outputs have distributed two units in total. Unlike an individual attention row, the accumulated tally does "}<strong>{"not"}</strong>{" sum to one. Here is a constructed three-position example; the entries are attention mass, not counts of distinct facts already expressed."}</Prose>

<NeuralTable caption={"Remember where a speech reader has been"} headers={[<>{"Read or tally"}</>,<>{"Source A"}</>,<>{"Source B"}</>,<>{"Source C"}</>,<>{"Total"}</>]} rows={[[<>{"First completed read"}</>,<>{"0.6"}</>,<>{"0.3"}</>,<>{"0.1"}</>,<>{"1"}</>],[<>{"Second completed read"}</>,<>{"0.2"}</>,<>{"0.5"}</>,<>{"0.3"}</>,<>{"1"}</>],[<>{"Coverage before the third read"}</>,<>{"0.8"}</>,<>{"0.8"}</>,<>{"0.4"}</>,<>{"2"}</>],[<>{"Third read"}</>,<>{"0.1"}</>,<>{"0.2"}</>,<>{"0.7"}</>,<>{"1"}</>],[<>{"Overlap: smaller of current share and old tally"}</>,<>{"0.1"}</>,<>{"0.2"}</>,<>{"0.4"}</>,<>{"0.7"}</>],[<>{"Coverage after the third read"}</>,<>{"0.9"}</>,<>{"1.0"}</>,<>{"1.1"}</>,<>{"3"}</>]]} />

<Prose>{"The third read sends most of its mass to C, which had received less attention. Its first 0.4 at C overlaps the old tally; the remaining 0.3 does not. The penalty is therefore 0.7. At the first read, the old tally is all zero and this overlap penalty is zero, whatever the first attention distribution. A penalty discourages repeated allocation; it does not ban it, prove factual coverage, or identify which meaning has already been verbalized. Location-aware features inspect the "}<strong>{"previous row's shape"}</strong>{"; coverage remembers the "}<strong>{"sum of all earlier rows"}</strong>{". They answer different questions and can coexist."}</Prose>

<H3>{"Cache what is fixed; measure what is expensive"}</H3>

<Prose>{"The encoder memory stays fixed while generating one output sequence. Cache projected keys once. Additive scoring's cache costs "}<InlineMath>{"O(Sd_hd_a)"}</InlineMath>{". Each read then projects a query in "}<InlineMath>{"O(d_sd_a)"}</InlineMath>{", scores positions in "}<InlineMath>{"O(Sd_a)"}</InlineMath>{", and combines values in "}<InlineMath>{"O(Sd_h)"}</InlineMath>{". General scoring can cache its "}<InlineMath>{"d_h\\to d_s"}</InlineMath>{" projection and use "}<InlineMath>{"O(Sd_s)"}</InlineMath>{" dot products per step."}</Prose>

<Prose>{"Across the output, source–target interactions have an "}<InlineMath>{"S T"}</InlineMath>{" factor. It becomes square only when source and target lengths are identified. Retaining every attention row uses "}<InlineMath>{"O(S T)"}</InlineMath>{" extra storage; sequential inference can keep one row at a time, plus source memory, keys and recurrent state. A teaching heatmap deliberately retains extra history."}</Prose>

<Prose>{"The recurrent state and input feeding still impose sequential decoder dependencies. Caching keys does not make those dependencies parallel. Beam search additionally needs a state, fed vector and prefix history per live candidate; it can share unchanged source memory. A source change requires a new cache, while corrected prefixes require replaying affected decoder states."}</Prose>

<Prose>{"Scaled dot-product attention divides the dot product by "}<InlineMath>{"\\sqrt d"}</InlineMath>{". Under independent, zero-mean, unit-variance query/key coordinates, the unscaled dot product has variance "}<InlineMath>{"d"}</InlineMath>{"; division keeps that variance near one under those assumptions. Learned recurrent states need not satisfy them. Inserting scaling into a fitted general-attention model changes its computation."}</Prose>

<Prose>{"Later multi-head layers add query/key/value projections, several reads and an output projection. A library multi-head module is not a drop-in reproduction of this decoder. Optimized dot-product kernels also do not automatically implement arbitrary additive scorers. Compare math, shapes, masks and outputs before choosing a fast kernel. Benchmark latency at the intended batch sizes, lengths, precision and device rather than inferring it from parameter counts. The "}<a href={"/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals"}>{"later self-attention lesson"}</a>{" develops those components."}</Prose>

<section className="lesson-ending lesson-ending--practice" data-lesson-ending="practice"><H2>{"9. Practice: calculate, diagnose and transfer"}</H2>

<Prose>{"Attempt each question before opening its hint or solution. Problems 1–5 test the core route; 6–8 use the optional branches."}</Prose>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"1. A changed question"}</H3>

<Prose>{"Let "}<InlineMath>{"q=(0,1)"}</InlineMath>{" with the three keys and values from section 2. Calculate scores, weights and context. Which memory gets the most weight?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"The query selects each key's second coordinate. Use the same weights for both value coordinates."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"Scores are "}<InlineMath>{"(0,1,0)"}</InlineMath>{", giving weights approximately "}<InlineMath>{"(0.211942,0.576117,0.211942)"}</InlineMath>{". B gets the most weight. The context is "}<InlineMath>{"(0.211942,1.364175)"}</InlineMath>{": first coordinate "}<InlineMath>{"2\\alpha_A-\\alpha_C"}</InlineMath>{", second "}<InlineMath>{"2\\alpha_B+\\alpha_C"}</InlineMath>{". Equal A/C scores do not imply equal values."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"2. A scorer that ignores its question"}</H3>

<Prose>{"A developer proposes "}<InlineMath>{"e_j=3q-2k_j+0.7"}</InlineMath>{" for scalar queries and keys. Will changing "}<InlineMath>{"q"}</InlineMath>{" change the attention weights? Suggest a score that can depend on both."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Factor quantities shared across source positions out of the softmax numerator and denominator."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"No: "}<InlineMath>{"\\exp(3q+0.7)"}</InlineMath>{" cancels. A product "}<InlineMath>{"qk_j"}</InlineMath>{", or a nonlinear score "}<InlineMath>{"\\tanh(3q-2k_j+0.7)"}</InlineMath>{", can change relative scores with the query. “Can” is deliberate: symmetry or saturation can still produce little change for a particular input."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"3. Where does the first read occur?"}</H3>

<Prose>{"A decoder starts in "}<InlineMath>{"s_0"}</InlineMath>{" and receives BOS. Someone computes "}<InlineMath>{"c_1=\\operatorname{Attention}(s_1,H)"}</InlineMath>{", then defines "}<InlineMath>{"s_1=\\operatorname{GRU}([\\operatorname{BOS};c_1],s_0)"}</InlineMath>{". Identify the problem and give two valid repairs."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Trace which value must exist first. A circular dependency is not yet an executable update rule."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"These equations require "}<InlineMath>{"s_1"}</InlineMath>{" to obtain "}<InlineMath>{"c_1"}</InlineMath>{" and vice versa, without defining a solver or intermediate state. A Bahdanau-style repair reads with "}<InlineMath>{"s_0"}</InlineMath>{", then updates using the context. A Luong-style repair updates from BOS first without the current context, then reads with "}<InlineMath>{"s_1"}</InlineMath>{" and combines the result in the output head. Input feeding can use the previous attentional vector, initially zero, without creating a current-step cycle."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"4. A padding bug hidden by the final word"}</H3>

<Prose>{"Two valid positions have scores "}<InlineMath>{"(0,0)"}</InlineMath>{" and scalar values "}<InlineMath>{"(2,4)"}</InlineMath>{". An unmasked zero-valued PAD position also has score 0. Calculate the correct and buggy contexts. Must their final greedy words differ?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Compare denominators containing two and three exponentials. Then distinguish continuous logits from their discrete argmax."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"The correct weights "}<InlineMath>{"(1/2,1/2)"}</InlineMath>{" give context 3. The buggy weights "}<InlineMath>{"(1/3,1/3,1/3)"}</InlineMath>{" give context 2. Logits can change while their largest entry stays the same, so identical greedy words do not rule out the bug. Mask invalid scores before normalizing."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"5. Plan a more specific comparison"}</H3>

<Prose>{"You want to learn whether additive versus general "}<strong>{"scoring alone"}</strong>{" affects this task. Is the two-architecture table a sufficient isolation? Describe a better experiment and what data it consumes."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"List changes besides the scorer: update order, decoder input width, output head and parameter count. Decide which must be held fixed for your narrower question."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"Keep the encoder, decoder order, context injection, output head, split, training schedule and decoding rule fixed; swap only the scorer. Report the remaining scorer parameter/computation differences. Predeclare seeds/metric, retain all runs and keep the rule baseline. This estimates behavior under one protocol, not a universal ordering. The already inspected partition remains development: repeated comparisons cannot become a fresh final test merely by changing the architecture name."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"6. Copying repeated unknown words"}</H3>

<Prose>{"Source tokens are "}<code>{"red blue red"}</code>{". Attention is "}<InlineMath>{"(0.15,0.25,0.60)"}</InlineMath>{", "}<InlineMath>{"p_{\\mathrm{gen}}=0.2"}</InlineMath>{", and the vocabulary is "}<code>{"blue:0.5, green:0.5"}</code>{". Compute the final probabilities, including "}<code>{"red"}</code>{"."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Combine the two red positions. Its vocabulary probability is zero; its copy probability is not."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"Copy mass is "}<code>{"red:0.75, blue:0.25"}</code>{". Final probabilities are "}<code>{"red:0.60, blue:0.30, green:0.10"}</code>{", summing to one. The extended output set includes source words. Taking only the largest red-position weight would discard its other occurrence."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"7. Does a Gaussian preserve a probability distribution?"}</H3>

<Prose>{"A window-normalized row is "}<InlineMath>{"(0.2,0.5,0.3)"}</InlineMath>{", with Gaussian multipliers "}<InlineMath>{"(0.5,1,0.5)"}</InlineMath>{". Compute the product's sum and its renormalized alternative. Why is a comment calling both “the same weighted average” wrong?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Multiply first. A second normalization changes every nonzero weight."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"The product is "}<InlineMath>{"(0.1,0.5,0.15)"}</InlineMath>{", sum 0.75. Renormalizing gives "}<InlineMath>{"(2/15,2/3,1/5)"}</InlineMath>{". With fixed values, the original context is 0.75 times the normalized context. Proportions match, but magnitudes and downstream logits need not. Declare the convention."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"8. A readable heatmap, an impossible streaming claim"}</H3>

<Prose>{"A speech system uses a bidirectional encoder over the complete recording and a local attention window. Its documentation says the window permits instant live transcription with no future audio. What information-path issue should be checked?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Attention is applied after encoding. Ask what a backward encoder state has already used."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"A memory can already depend on future audio. Restricting a subsequent read to nearby positions cannot remove that dependency. A streaming design must declare encoder lookahead, chunk/buffering policy, permitted memory and output latency as well as the attention movement rule. It may need a causal or limited-lookahead encoder and an online alignment mechanism. A narrow visible window proves none of those properties by itself."}</Prose>

</details></div></section>

<section className="lesson-ending lesson-ending--resources" data-lesson-ending="resources"><H2>{"10. Continue the route and choose another explanation"}</H2>

<Prose>{"You are ready to continue when you can calculate a masked read, trace both decoder schedules, explain how output loss trains a score, and distinguish a changed attention picture from a changed output. Copying and speech can remain optional return points."}</Prose>

<Prose>{"Next is "}<a href={"/learn/path/full-curriculum/long-context-sequence-models-transformer-xl-griffin-perceiver?module=deep-learning-fundamentals"}>{"Long-Context Sequence Models: Transformer-XL, Griffin and Perceiver"}</a>{". It asks how access and computation change when keeping or reading all memory becomes expensive. It introduces additional mechanisms locally; the later "}<a href={"/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals"}>{"Self-Attention & Multi-Head Attention"}</a>{" provides their dedicated treatment."}</Prose>

<Prose>{"Useful alternate routes:"}</Prose>

<ul><li>{""}<strong>{"A visual overview:"}</strong>{" "}<a href={"https://distill.pub/2016/augmented-rnns/"}>{"Olah and Carter, Attention and Augmented Recurrent Neural Networks"}</a>{", especially “Attentional Interfaces,” shows how a changing read connects a reader and writer. The article's attention section was reviewed; it is a classic mechanism explanation, not current library documentation."}</li><li>{""}<strong>{"Lookup before algebra:"}</strong>{" "}<a href={"https://d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html"}>{"D2L, Queries, Keys, and Values"}</a>{" starts from retrieval and connects it to weighted pooling. Use it alongside section 2; our weather and vector examples are independently constructed."}</li><li>{""}<strong>{"Visual next step into transformers:"}</strong>{" "}<a href={"https://www.3blue1brown.com/lessons/attention/"}>{"3Blue1Brown, Attention in transformers, step-by-step"}</a>{" (2024) explains the desired contextual behavior before the matrix operations, with an accompanying "}<a href={"https://www.youtube.com/watch?v=eMlx5fFNoYc"}>{"video"}</a>{". The article and its diagrams were reviewed; the complete video was not watched. It teaches transformer self-attention, so return to our decoder timelines when comparing it with Bahdanau or Luong."}</li></ul>

<ul><li>{""}<strong>{"Textbook with tensor code:"}</strong>{" "}<a href={"https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html"}>{"D2L 1.0.3, Attention Scoring Functions"}</a>{" covers masks, batch matrix products, dot and additive scores. Use it after section 2 to connect vector calculations with batched code. Its helpers/conventions differ from this program; compare definitions before mixing snippets."}</li><li>{""}<strong>{"Recurrent attention chapter:"}</strong>{" "}<a href={"https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html"}>{"D2L 1.0.3, Bahdanau Attention"}</a>{" connects stored outputs, valid lengths and decoder queries. Its translation experiment is separate from our inflection measurements."}</li><li>{""}<strong>{"Video and notes:"}</strong>{" "}<a href={"https://www.youtube.com/watch?v=XXtpJxZBa2c"}>{"Stanford Online CS224N Lecture 8: Neural Machine Translation, Seq2seq and Attention"}</a>{", with "}<a href={"https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes06-NMT_seq2seq_attention.pdf"}>{"official 2019 notes"}</a>{". A spoken alternative for following the timelines. Lecture identity and companion notes were checked; the full video was not watched during authoring. Use the corrected primary equations here for nonlinear concat and exact input-feeding conventions."}</li><li>{""}<strong>{"Original motivation:"}</strong>{" "}<a href={"https://arxiv.org/pdf/1409.0473"}>{"Bahdanau, Cho and Bengio"}</a>{", especially sections 3/5 and appendix A. Its historical translation results do not establish a universal input-length boundary."}</li><li>{""}<strong>{"Scorers, windows and input feeding:"}</strong>{" "}<a href={"https://arxiv.org/pdf/1508.04025v5"}>{"Luong, Pham and Manning, arXiv v5"}</a>{", sections 3/4. Use its explicit nonlinear concat formula. Its local-p multiplication differs from our explicitly renormalized alternative."}</li><li>{""}<strong>{"Applications:"}</strong>{" "}<a href={"https://arxiv.org/pdf/1704.04368"}>{"Pointer-generator networks"}</a>{", "}<a href={"https://arxiv.org/pdf/1506.07503"}>{"location-aware speech attention"}</a>{", and "}<a href={"https://arxiv.org/pdf/1508.01211"}>{"Listen, Attend and Spell"}</a>{". Start at the mechanism sections cited above. Their full experimental reproduction is a separate project."}</li></ul>

<Prose>{"The "}<a href={"/learn-code/attention-mechanism-bahdanau-luong/data-provenance.md"}>{"provenance"}</a>{", "}<a href={"/learn-code/attention-mechanism-bahdanau-luong/attentive-inflection.py"}>{"executed program"}</a>{", "}<a href={"/learn-code/attention-mechanism-bahdanau-luong/calculated-inputs.json"}>{"actual trained results"}</a>{", "}<a href={"/learn-code/attention-mechanism-bahdanau-luong/attention-calculations.py"}>{"constructed calculations"}</a>{" and "}<a href={"/learn-code/attention-mechanism-bahdanau-luong/mechanics-results.json"}>{"saved-model traces"}</a>{" make the page's numbers inspectable."}</Prose></section>
  </div>,
};
