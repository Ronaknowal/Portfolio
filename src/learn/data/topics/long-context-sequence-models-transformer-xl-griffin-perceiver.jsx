// Generated from the active concept-intuition manuscript; revision-4 evidence remains historical.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { MemoryCacheLab, RecurrentMemoryLab, LatentWorkspaceLab, LongContextProgram } from '../../components/lesson-labs/LongContextLabs.jsx';
import { LongContextTrajectoryLab } from '../../components/lesson-labs/LongContextTrajectoryLab.jsx';
import { MemoryWorkspacesFigure, MaskedReadFigure, AttentionShapesFigure, SegmentLayersFigure, PositionAddressFigure, RetentionFigure, GriffinPathsFigure, PerceiverReadFigure, PositionAttachmentFigure, OutputQueriesFigure, CausalLatentsFigure, TrajectoryArchitectureFigure, TrajectoryEvidenceFigure, DetachedMemoryFigure, MemoryBudgetsFigure, AffineScanFigure } from '../../components/lesson-labs/LongContextFigures.jsx';
import { EntranceMessageFigure, WeightedReadSharesFigure, SilentStepGateFigure, PathMeanCollisionFigure } from '../../components/lesson-labs/LongContextIntuitionFigures.jsx';
export default {
  title: 'Long-Context Sequence Models (Transformer-XL, Griffin, Perceiver)',
  readTime: '~80 min read + experiments and practice',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson long-context-lesson">
    <LessonIntro prerequisites="Basic algebra and the preceding attention lesson help. We rebuild weighted reads, stored state, queries, masks and array shapes through small examples before using them." sections={[["1-three-kinds-of-memory-are-different-objects","1. Three kinds of memory are different objects"],["2-a-small-attention-operation-you-can-calculate","2. A small attention operation you can calculate"],["3-transformer-xl-continue-across-segment-boundaries","3. Transformer-XL: continue across segment boundaries"],["4-griffin-keep-a-recurrent-summary-and-consult-nearby-detail","4. Griffin: keep a recurrent summary and consult nearby detail"],["5-perceiver-put-deep-computation-in-a-smaller-workspace","5. Perceiver: put deep computation in a smaller workspace"],["6-which-perceiver-is-allowed-to-generate-left-to-right","6. Which Perceiver is allowed to generate left to right?"],["7-run-the-mechanisms-then-try-real-movement-data","7. Run the mechanisms, then try real movement data"],["8-deeper-gradients-budgets-and-a-useful-evaluation-plan","8. Deeper: gradients, budgets and a useful evaluation plan"],["9-practice-implement-a-change-and-explain-its-effect","9. Practice: implement a change and explain its effect"],["10-references-another-way-to-learn-it","10. References & another way to learn it"]]}>A model can read a fact and still lose access to it later. Follow what each kind of memory keeps, calculate how it changes, and then build and inspect the mechanisms yourself.</LessonIntro>
<Prose>{"A note at the beginning of a document says, “The backup entrance is on the east side.” Much later, someone asks which entrance to use. A model must carry that earlier information forward or be able to consult it again. Merely accepting the whole document as input does not tell us whether it can answer."}</Prose>

<Prose>{"Imagine reading the document through a small opening that shows only the last two messages. When the question arrives, the entrance note has disappeared. Reading the visible question more carefully cannot recover the missing direction. A useful system needs a deliberate way to carry information across that boundary."}</Prose>

<EntranceMessageFigure />

<Prose>{"Look at what disappeared in the figure: the evidence, not the question. We will compare three ways of preserving useful evidence. "}<strong>{"Transformer-XL"}</strong>{" keeps earlier records available for later consultation. "}<strong>{"Griffin"}</strong>{" combines an updated summary with detailed access to the recent past. "}<strong>{"Perceiver"}</strong>{" reads a large available input into a smaller workspace where most subsequent processing happens. These are architectural choices about information flow; learning determines what each system actually preserves."}</Prose>

<Prose>{"The same choice matters when a recorded hand movement must be identified as a line, curve or zigzag. Keeping only the path's average location is cheap, but it can erase its shape. Later we will train and inspect a small model on real hand trajectories, so the discussion ends with a concrete decision rather than a list of model names."}</Prose>

<Prose>{""}<strong>{"How to read this lesson."}</strong>{" Sections 1–2 establish the memory objects and a single weighted read. Sections 3–6 explain one architecture at a time: the problem it solves, a small example, the mechanism and then the formal detail. Try each investigation after its worked example; changes appear immediately. Section 7 connects those operations to complete scratch and library programs and real data. Section 8 is a second-pass route through gradients, scaling and evaluation. Finish with the changed problems in section 9. You can take the architecture sections in separate sittings without losing the common question: "}<strong>{"what remains available when the next computation needs it?"}</strong>{""}</Prose>

<Prose>{"The preceding "}<a href={"/learn/path/full-curriculum/attention-mechanism-bahdanau-luong?module=deep-learning-fundamentals"}>{"Bahdanau and Luong attention lesson"}</a>{" introduced attention as a way for a decoder to consult different encoder states. We will refresh that operation locally and then change what is available to consult. The later "}<a href={"/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals"}>{"self-attention"}</a>{", "}<a href={"/learn/path/full-curriculum/transformer-block-architecture?module=deep-learning-fundamentals"}>{"transformer-block"}</a>{" and "}<a href={"/learn/path/full-curriculum/positional-encodings-sinusoidal-learned-rope-alibi?module=deep-learning-fundamentals"}>{"positional-encoding"}</a>{" lessons develop those building blocks more fully. No knowledge of their implementations is required here."}</Prose>

<H2>{"1. Three kinds of memory are different objects"}</H2>

<H3>{"Keep the record, update a summary, or make working notes"}</H3>

<Prose>{"Start with an ordinary task: record five temperatures and answer questions about them. If the question is “What was the second reading?”, retaining the five readings is a direct solution. If the question is “What is the running average?”, a sum and a count are enough; the individual readings can be discarded. Those two tasks need different kinds of memory."}</Prose>

<Prose>{"Neural models usually store a "}<strong>{"representation"}</strong>{" of an input: a list of feature values used by later computations. Such a list is called a "}<strong>{"vector"}</strong>{". Its entries need not be readable facts like “east”; they are numbers learned for the task. The crucial question is whether each earlier position still has its own accessible vector or has been combined into something else."}</Prose>

<Prose>{"There are three useful arrangements:"}</Prose>

<ul><li>{""}<strong>{"Keep separate records."}</strong>{" The next question can consult different earlier positions separately. A bounded collection of such stored representations is often called a "}<strong>{"cache"}</strong>{"."}</li><li>{""}<strong>{"Update a summary."}</strong>{" Each new input modifies the same fixed-size list of numbers. This repeatedly updated list is the "}<strong>{"recurrent state"}</strong>{". A sum and a count are a simple hand-designed example; a neural state learns what to track."}</li><li>{""}<strong>{"Make a small set of working notes."}</strong>{" Keep the large input available, but collect selected information into a few working vectors. These are "}<strong>{"latents"}</strong>{": internal representations inferred for the task. A collection of them is a latent array. Further computation happens among those vectors, and they may consult the original input again."}</li></ul>

<Prose>{"The figure below follows the same five inputs. In the middle arrangement, watch the state being updated: there is no separate slot from which to retrieve input 0. On the right, notice that the input bank remains available outside the smaller workspace. A latent bottleneck therefore differs from a stream that has already discarded its old inputs."}</Prose>

<MemoryWorkspacesFigure />

<H3>{"“Long context” can mean four different things"}</H3>

<Prose>{"Returning to the entrance note, accepting every message is only the first requirement. The answering computation also needs an information path from that note, enough capacity to retain what matters, and learned behavior that uses it. Separate these questions when reading a model description:"}</Prose>

<NeuralTable caption={"“Long context” can mean four different things"} headers={[<>{"Quantity"}</>,<>{"Question it answers"}</>]} rows={[[<>{"Input length"}</>,<>{"How many positions are supplied to a computation?"}</>],[<>{"Direct attention span"}</>,<>{"Which earlier positions can this query consult individually at this layer?"}</>],[<>{"State or cache capacity"}</>,<>{"What representations are stored between steps or segments?"}</>],[<>{"Useful dependency length"}</>,<>{"How far back does the model actually use information successfully on this task?"}</>]]} />

<Prose>{"A recurrent model can process an arbitrarily long stream using bounded state while losing a particular early fact. A finite-window attention layer may receive a state that already summarizes older material. A latent model can read all pixels of an image without preserving every detail in its bottleneck. These are mechanisms to evaluate, rather than contradictory claims about a single “context length.”"}</Prose>

<Prose>{"Our document example needs selective recall. A temperature average is naturally compressible into two numbers; many independently requested facts are harder to compress without losing distinctions. This is why there is no single best memory object for every task. Next we need the operation that allows a question to consult the records that remain."}</Prose>

<H2>{"2. A small attention operation you can calculate"}</H2>

<H3>{"First combine the information; then explain how the weights arise"}</H3>

<Prose>{"Attention produces a task-dependent weighted mixture. For the entrance question, the current computation should give more weight to an entrance-related record than to a lunch announcement. The comparison part of a record is its "}<strong>{"key"}</strong>{"; the information it contributes is its "}<strong>{"value"}</strong>{". The request is the "}<strong>{"query"}</strong>{". Separate those roles: a record can be easy to match while carrying a small value, or hard to match while carrying a large one."}</Prose>

<Prose>{"For a calculation you can follow exactly, replace word features with three scalar readings: 2, 4 and 8. Suppose the query assigns their keys relative supports of 2, 1 and 3. There are six units of support in total. The first reading receives 2/6 of the mixture, the second 1/6 and the third 3/6. Multiply each value by its share and add: (2×2 + 1×4 + 3×8)/6 = 32/6, about 5.333. Larger support means more influence on this read, not a claim that the source is more factually reliable."}</Prose>

<WeightedReadSharesFigure />

<Prose>{"Read the figure from the shares to the contributions. The third reading owns half the mixture, so it contributes 4 to the final answer. The first two together contribute 4/3. This is the whole weighted-read mechanism. Neural attention adds a learned way to choose those shares."}</Prose>

<H3>{"Scores turn a question into a mixture"}</H3>

<Prose>{"An attention query is a vector expressing what information is requested. A learned "}<strong>{"linear projection"}</strong>{" multiplies a representation by a weight matrix to produce its query, key or value vector. The query and each key receive a comparison score. A "}<strong>{"dot product"}</strong>{" computes that score by multiplying matching coordinates and adding them; a larger result favors that key. Softmax then turns all the allowed scores into nonnegative weights summing to one."}</Prose>

<Prose>{"Here is the same recipe in notation, one step at a time: sⱼ is a comparison score; αⱼ is that record's share of the read; o is the resulting mixture. For one query q and available pairs (kⱼ,vⱼ):"}</Prose>

<div className="neural-equation"><MathBlock>{"s_j=\\frac{q^\\top k_j}{\\sqrt{d_k}}+b_j,\n\\qquad\n\\alpha_j=\\frac{\\exp(s_j)}{\\sum_{r\\in\\mathcal A}\\exp(s_r)},\n\\qquad\no=\\sum_{j\\in\\mathcal A}\\alpha_jv_j."}</MathBlock></div>

<Prose>{"Here dₖ counts query/key coordinates; bⱼ optionally changes a score using position; and 𝒜 contains the positions the query may use. The square-root factor controls score scale as the number of coordinates changes. For our scalar example dₖ=1, so the factor is 1. The sum sign means “add this term for every allowed record.”"}</Prose>

<Prose>{"We can now generate exactly the supports in the picture. Let q=1 and use keys ln 2, 0 and ln 3 with no positional bias. The natural logarithm is the inverse of exponentiation, so exp(ln 2)=2, exp(0)=1 and exp(ln 3)=3. Softmax normalizes these supports, giving the weighted read we already calculated:"}</Prose>

<div className="neural-equation"><MathBlock>{"o=\\frac{2(2)+1(4)+3(8)}{2+1+3}=\\frac{16}{3}."}</MathBlock></div>

<H3>{"A forbidden record gets no share at all"}</H3>

<Prose>{"Suppose the third reading has not happened yet. A prediction made now may use 2 and 4, but cannot consult the future 8. We remove the third record before assigning shares. The remaining supports total 3; the weights become 2/3 and 1/3, and the answer is 8/3. Setting the third *value* to zero while keeping its support in the denominator would instead give 4/3. That incorrect answer shows why a "}<strong>{"mask"}</strong>{" must exclude a record from the entire read, not merely erase its visible value."}</Prose>

<MaskedReadFigure />

<Prose>{"For many queries, do one such read for each request. Two questions about five records require ten comparison scores and produce two answers. In a matrix, rows simply keep those questions separate. Arrange queries as Q with shape L×dₖ, keys as K with shape N×dₖ and values as V with shape N×dᵥ. QKᵀ contains L×N scores; weighting V produces L×dᵥ outputs. "}<strong>{"The number of output positions comes from the queries."}</strong>{" The shape diagram below is bookkeeping for the individual reads you just followed, not a new operation."}</Prose>

<AttentionShapesFigure />

<Prose>{"For next-token prediction, the representation at position i may use inputs through i and predict token i+1. A "}<strong>{"causal mask"}</strong>{" permits j≤i. For a complete-trajectory classification task, all measured points are already available before the classification is requested, so a noncausal read is appropriate. The mask follows the task's information boundary."}</Prose>

<H2>{"3. Transformer-XL: continue across segment boundaries"}</H2>

<H3>{"3.1 Why separate chunks forget"}</H3>

<Prose>{"Processing a long document in chunks is attractive: each chunk fits a manageable amount of work into memory. But imagine a chunk ending “Use the backup entrance on the east…” and the next beginning “…side.” Starting the second chunk with no earlier information has cut a connected thought in two. This is "}<strong>{"context fragmentation"}</strong>{". It can harm understanding even when the needed words are close together."}</Prose>

<Prose>{"The repair is to carry some already-computed records into the next chunk. Transformer-XL calls a chunk a "}<strong>{"segment"}</strong>{". When reading a new segment, its current positions ask questions; earlier stored representations join the current representations as possible donors. A query near the start can now consult the previous segment. The "}<a href={"https://research.google/blog/transformer-xl-unleashing-the-potential-of-attention-models/"}>{"authors' illustrated explanation"}</a>{" makes this segment boundary the central problem; the exact layer construction comes after our small trace below."}</Prose>

<Prose>{"That earlier memory is an additional set of allowed sources. It does not permit a query to look ahead within the new segment. We still apply the causal rule from section 2."}</Prose>

<H3>{"3.2 Follow a complete cache example"}</H3>

<Prose>{"Let the five records have values [6,1,8,2,0]. Record 0 is a strong match for our question: give it relative support 9, compared with support 1 for every other record. With scalar query 1, keys [ln 9,0,0,0,0] produce exactly those supports. This constructed example isolates access to a useful record; it is not a trained text model."}</Prose>

<Prose>{"Process segments [0,1], [2,3], then [4]. Follow a cache of length 2 through the boundary. After the first segment it holds records 0 and 1. After the second, keeping only the newest two replaces them with records 2 and 3. When query 4 arrives, record 0 is gone even though it was read earlier. Cache length controls what can still be consulted directly."}</Prose>

<Prose>{"At the final query, compare three memory lengths:"}</Prose>

<NeuralTable caption={"3.2 Follow a complete cache example"} headers={[<>{"Retained positions before the final segment"}</>,<>{"Legal values"}</>,<>{"Output"}</>]} rows={[[<>{"None, M=0"}</>,<>{"Current value 0"}</>,<>{"0"}</>],[<>{"Positions 2 and 3, M=2"}</>,<>{"8, 2, 0 with equal weights"}</>,<>{"10/3"}</>],[<>{"Positions 0–3, M=4"}</>,<>{"6, 1, 8, 2, 0 with weights proportional to 9,1,1,1,1"}</>,<>{"65/13=5"}</>]]} />

<Prose>{"The output is a soft combination, not a lookup returning exactly 6. The larger cache makes the strong matching key available again. With M=4, every query in this five-position example sees the same legal keys as the full causal calculation, so their outputs agree. This equality holds here because the projections and positions are fixed and every required key remains available."}</Prose>

<Prose>{"The live record strip starts at the middle row of that comparison: memory length 2 and query position 4. Increase retained memory to 4. R0 becomes available, and the answer moves from 10/3 to 5. Notice that its stored value stayed 6; the change came from access and normalization. Next inspect query position 1, edit a later record's value, and observe the unchanged answer. These two edits separate missing history from correctly excluded future information."}</Prose>

<MemoryCacheLab />

<H3>{"3.3 Put the cache into a layered model"}</H3>

<Prose>{"A neural "}<strong>{"layer"}</strong>{" turns input representations into new representations. A stack of layers repeats that transformation. The records entering one layer may therefore already contain information gathered by earlier layers. In Transformer-XL, an upper-layer read uses the lower layer's current representations together with retained lower-layer representations from the preceding segment."}</Prose>

<Prose>{"In the diagram, first follow the solid forward path across the segment boundary. It answers “which values can the current computation use?” Then look at the stopped training path. Training can use those old values while treating their earlier creation as outside the current gradient calculation. This operation is called "}<strong>{"detaching"}</strong>{" or "}<code>{"stopgrad"}</code>{"; it changes credit assignment, not the value stored in memory. Section 8 calculates the difference explicitly."}</Prose>

<SegmentLayersFigure />

<Prose>{"For layer ℓ in segment s, let Hₛ⁽ℓ⁻¹⁾ contain L current rows of width d and let Mₛ⁽ℓ⁻¹⁾ contain M retained rows of that same width. The following expression is just “put old and current donor rows together; ask questions only for the current rows”:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\widetilde H_s^{(\\ell-1)}\n=[\\operatorname{stopgrad}(M_s^{(\\ell-1)});H_s^{(\\ell-1)}],\n\\quad\nQ=H_s^{(\\ell-1)}W_Q,\n\\quad\nK=\\widetilde H_s^{(\\ell-1)}W_K,\n\\quad\nV=\\widetilde H_s^{(\\ell-1)}W_V."}</MathBlock></div>

<Prose>{"The semicolon joins rows, producing M+L donors. There are still L queries, so attention produces L current outputs. After the segment, retain the latest M appropriate hidden states for the next segment. The W matrices are the learned projections that create the query, key and value roles introduced in section 2."}</Prose>

<Prose>{"The "}<a href={"https://arxiv.org/pdf/1901.02860"}>{"Transformer-XL paper, sections 3.2–3.3"}</a>{" defines this layerwise recurrence and its relative-position construction. Detached memory still affects the current prediction, and parameters projecting that memory into current keys and values still receive gradients; only the history that created the stored tensor is detached."}</Prose>

<Prose>{"A real multilayer Transformer-XL can pass older information through representations that were themselves contextualized. Its indirect dependency paths differ from a single layer's list of visible positions. With the paper's one-segment memory construction, crossing a segment boundary also moves through a layer, giving a depth-dependent receptive field. A finite network and cache do not imply unlimited exact recall."}</Prose>

<H3>{"3.4 A reused position needs a coherent address"}</H3>

<Prose>{"Imagine numbering positions within every segment as 0,1,2,3. A cached token from local position 1 and a new token at local position 1 now carry the same absolute-position label. Reusing those labels creates ambiguity about temporal relationships."}</Prose>

<Prose>{"A relative position asks, “How far is this key from the query?” A query at global position 5 sees keys at positions 1 and 5 at distances 4 and 0. Shifting all three global positions by 100 preserves those distances. Resetting segment-local counters does not."}</Prose>

<PositionAddressFigure />

<Prose>{"Keeping records creates a second engineering problem: how to tell the old “position 1” from the new “position 1.” The distance ruler above fixes the meaning of “four positions ago” across segment boundaries. For the first pass, this is the important idea. The exact Transformer-XL score below separates content matching from distance preferences."}</Prose>

<Prose>{"An attention "}<strong>{"head"}</strong>{" is one learned set of query/key/value projections; multiple heads can ask different questions. Transformer-XL uses separate content and relative-position projections. One head's score can be written"}</Prose>

<div className="neural-equation"><MathBlock>{"s_{ij}=q_i^\\top k_j+q_i^\\top r_{i-j}\n+u^\\top k_j+v^\\top r_{i-j},"}</MathBlock></div>

<Prose>{"with scaling applied according to the head implementation. The first term matches content; the second makes preferred distance depend on the current query; the third is a learned global content preference; the fourth is a learned global distance preference. Here r is the projected relative-position vector, and u and v are learned vectors. These v and r symbols are score parameters; vⱼ in section 2 denoted an attention value."}</Prose>

<Prose>{"To separate the two distance terms, temporarily set both content keys to zero. Use a constructed one-coordinate head whose projected distance features are +1 for a near key and −1 for a far key, with global distance parameter v=0.5. This is a small numerical example of the score, not Transformer-XL's actual learned position features."}</Prose>

<NeuralTable caption={"3.4 A reused position needs a coherent address"} headers={[<>{"Contribution"}</>,<>{"Near key: r=+1"}</>,<>{"Far key: r=−1"}</>]} rows={[[<>{"Global distance term v r, for either query"}</>,<>{"+0.5"}</>,<>{"−0.5"}</>],[<>{"Query-dependent distance term q r when q=+1"}</>,<>{"+1"}</>,<>{"−1"}</>],[<>{"Total distance score when q=+1"}</>,<>{"+1.5"}</>,<>{"−1.5"}</>],[<>{"Query-dependent distance term q r when q=−1"}</>,<>{"−1"}</>,<>{"+1"}</>],[<>{"Total distance score when q=−1"}</>,<>{"−0.5"}</>,<>{"+0.5"}</>]]} />

<Prose>{"The global term always favors the near key in this example. Changing the query reverses the other term strongly enough that the far key now wins. This is the benefit of a query-dependent distance preference: the current question can call for different temporal relationships. A fixed recency penalty cannot reverse its preference that way. In the full score, content terms can change the winner again; relative distance alone does not determine attention. "}<a href={"https://arxiv.org/pdf/1901.02860"}>{"Transformer-XL, §3.3"}</a>{""}</Prose>

<Prose>{"Our downloadable scalar calculation can instead add −β(i−j) to the score. It is a transparent recency-bias exercise, "}<strong>{"not Transformer-XL's full positional formula"}</strong>{". It lets you observe how recency can compete with a content match. The later positional-encoding lesson explains sinusoidal features, RoPE and other constructions separately."}</Prose>

<H2>{"4. Griffin: keep a recurrent summary and consult nearby detail"}</H2>

<H3>{"4.1 Why ignoring a new input can still erase an old one"}</H3>

<Prose>{"A cache grows by keeping more records. A recurrent model has another option: keep updating the same small workspace. Imagine tracking whether an important event occurred while many uninformative events pass by. A state value of .6 might represent the event's current influence. If each update retains only 80% of the previous state, the next value is .48—even when the new input is zero. Silence does not automatically mean “leave memory alone.”"}</Prose>

<Prose>{"Separate the two contributions to an update: "}<strong>{"keep some of the old state, then add some of the new signal"}</strong>{". Write the old state as hₜ₋₁, the incoming number as xₜ and the new state as hₜ. A simple update is"}</Prose>

<div className="neural-equation"><MathBlock>{"h_t=a h_{t-1}+b x_t."}</MathBlock></div>

<Prose>{"Here a controls retention and b controls injection. Start from zero, set a=.8 and b=.6, and send one input of 1 followed by five zeros. The states are .6, .48, .384, .3072, .24576 and .196608. The first event is still influential, but repeated multiplication by .8 has weakened it. Unlike the cache, there is no separate old record to reopen."}</Prose>

<Prose>{"Griffin uses learned controls called "}<strong>{"gates"}</strong>{" to change this behavior. An input gate controls the new signal; a recurrence gate controls how much old state survives. To distinguish them, keep the current input at zero. Closing its input gate does nothing: the new signal was already zero. Holding the old state requires changing retention instead."}</Prose>

<SilentStepGateFigure />

<Prose>{"In the figure, compare the first two rows before the third. The first two have different input-gate settings but identical output. Their shared .12 loss happened on the old-state path. That is the practical reason for having a separate recurrence gate."}</Prose>

<H3>{"4.2 How the two gates set the update"}</H3>

<Prose>{"Griffin's "}<strong>{"real-gated linear recurrent unit"}</strong>{", or "}<strong>{"RG-LRU"}</strong>{", makes those gates depend on the current input. A sigmoid turns a learned score into a number between 0 and 1. A learned base a between 0 and 1 sets a coordinate's decay behavior; the recurrence gate rₜ changes its effective retention to aₜ=a^(c·rₜ). The scale c is 8 in the paper. With a=.8 and rₜ=1/8, retention is .8. As rₜ approaches zero, retention approaches 1."}</Prose>

<Prose>{"The incoming multiplier is √(1−aₜ²). Thus keeping more old state also reduces the scale of the injected signal. With retention .8, that multiplier is √(.36)=.6. With retention approaching 1, it approaches zero. These are coupled choices, not two arbitrary coefficients that can both be turned up without consequence."}</Prose>

<Prose>{"We can now write the full vector update. A vector contains several state coordinates; the symbol ⊙ means multiply matching coordinates separately. σ is the sigmoid, W and b are learned gate weights and offsets, and iₜ is the input gate:"}</Prose>

<div className="neural-equation"><MathBlock>{"r_t=\\sigma(W_r x_t+b_r),\\qquad\ni_t=\\sigma(W_i x_t+b_i),\\qquad\na_t=a^{c r_t},"}</MathBlock></div>

<div className="neural-equation"><MathBlock>{"h_t=a_t\\odot h_{t-1}\n+\\sqrt{1-a_t^2}\\odot(i_t\\odot x_t)."}</MathBlock></div>

<Prose>{"Read the last line as two paths meeting: aₜ⊙hₜ₋₁ retains old state; √(1−aₜ²)⊙(iₜ⊙xₜ) admits new signal. The input gate iₜ controls only the signal inside that second path. The recurrence gate rₜ changes both retention and its normalization. This is the update in "}<a href={"https://arxiv.org/pdf/2402.19427"}>{"Griffin, section 2.4"}</a>{"."}</Prose>

<Prose>{"With a=.8 and rₜ=1/8, the effective decay is .8. With rₜ near zero, aₜ approaches 1 and the injection multiplier approaches zero. The unit can preserve a state across an uninformative interval instead of repeatedly replacing it. In the exact limiting case rₜ=0, it holds the state unchanged. Finite sigmoid logits approach that endpoint but do not produce it exactly."}</Prose>

<RetentionFigure />

<Prose>{"Follow the two curves above through the zero-input interval. Both start at .6. Their later difference comes entirely from how much old state survives; no later event adds a signal. The plotted values come from the same recurrence as the live investigation below."}</Prose>

<Prose>{""}<strong>{"Why the square root?"}</strong>{" It balances the scale of old and new contributions under a useful simplifying assumption. If hₜ₋₁ and the gated input each have variance 1 and are uncorrelated, the combined variance is aₜ²+(1−aₜ²)=1 when the coefficient is fixed. Learned, input-dependent signals need not satisfy those assumptions. This calculation explains the design rather than asserting every hidden coordinate has variance exactly one."}</Prose>

<Prose>{"For the six-step impulse above, use r₁=1/8 and r₂…r₆=.001. The final state is about .594668, close to the first state's .6. If we only set the later input gates to zero while leaving rₜ=1/8, the final state remains .196608. The inputs were already zero; stopping input admission did not stop the recurrence from decaying."}</Prose>

<Prose>{"Try the live sequence in this order: compare the initial ordinary decay with “Near-hold after first event.” Then reset the recurrence before selecting “Close later input gates only.” The first edit changes the final state; the second does not. Next add a later nonzero event and adjust its input gate; now that gate has an incoming signal to control. The retained and injected terms show exactly where each change enters."}</Prose>

<RecurrentMemoryLab />

<H3>{"4.3 Give the summary a nearby detailed view"}</H3>

<Prose>{"The recurrence solves one problem—carrying an influence forward in bounded state—but its summary cannot expose every earlier record separately. Griffin combines it with "}<strong>{"local sliding-window attention"}</strong>{": a read that keeps nearby positions individually accessible. Recent exact detail and older compressed influence can therefore take different routes."}</Prose>

<Prose>{"The published Griffin construction repeats two recurrent blocks followed by one local-attention block; the usual attention window in that study is 1,024 tokens. In the next figure, follow the state through time separately from the row of recent key/value records. An old event can affect the state without still occupying a directly readable attention slot."}</Prose>

<GriffinPathsFigure />

<Prose>{"To connect this memory picture to the implementation, open one recurrent block. It projects each input into two branches. One branch first mixes a few recent timesteps with a "}<strong>{"causal depthwise temporal convolution"}</strong>{": each channel has its own short filter, which uses the present and past rather than the future. RG-LRU then carries that filtered signal through time. The other branch supplies a nonlinear gate. Multiplying the branches controls which processed features pass on; a final projection restores the model width."}</Prose>

<Prose>{"Residual connections add a block's update to its incoming representation, normalization controls feature scale, and a gated feed-forward block transforms features at each position. These supporting pieces do not change which earlier records are directly stored. The "}<a href={"https://developers.googleblog.com/gemma-explained-recurrentgemma-architecture/"}>{"official RecurrentGemma walkthrough"}</a>{" illustrates how the branches and dimensions fit together. Its released model settings and Griffin's paper settings should be read separately; “local attention” does not imply one universal window size."}</Prose>

<Prose>{"A local-attention cache is bounded once its window is full. A recurrent block also needs its state and the short convolution history. Dense projection and feed-forward weights remain part of the model. “Memory independent of stream length” refers to the bounded recurrent/cache state under a fixed architecture, not to the entire training computation or arbitrary batch size."}</Prose>

<Prose>{"The next lesson studies state-space models in depth. RG-LRU is not presented in its source as a discretization of a continuous-time system, and this chapter's gated recurrence should not be silently substituted for a Mamba update. Similar diagrams can conceal different equations."}</Prose>

<H2>{"5. Perceiver: put deep computation in a smaller workspace"}</H2>

<H3>{"5.1 Why one average can lose the answer"}</H3>

<Prose>{"Sometimes the whole input is already available: an image, a recorded sound or a hand's measured path. There is no need to predict before its later elements arrive. The problem is the amount of computation needed to reason about the full collection repeatedly."}</Prose>

<Prose>{"A tempting shortcut is to replace a path with its average coordinate. But consider five evenly spaced horizontal positions from 0 to 1. In path A, all heights are .5. In path B, the heights are [.5,.75,0,.75,.5]. Both average to (.5,.5), even though one is straight and the other changes direction. If a classifier receives only that average, the lost shape is unavailable to it."}</Prose>

<PathMeanCollisionFigure />

<Prose>{"Look at the path before the center mark. Their centers agree; their geometries do not. This is the reason to learn several selective summaries rather than automatically average everything. It does not prove a particular learned model will succeed—we will test that question on real trajectories in section 7."}</Prose>

<H3>{"5.2 Learn a few questions instead of processing every input pair"}</H3>

<Prose>{"Perceiver makes a smaller working array of learned vectors called "}<strong>{"latents"}</strong>{". Think of each starting latent as a trainable request for information. Its initial numbers are model parameters shared across examples. After it reads an input, its updated numbers depend on that particular example. The model learns useful requests through the training task, not through hand-written names such as “find a curve.”"}</Prose>

<Prose>{"Each latent asks a query of the original input records, using the attention operation from section 2. If there are N latent queries and T input records, there are N×T comparison scores and N resulting vectors. This is "}<strong>{"cross-attention"}</strong>{" because queries and donor records come from different arrays. The resulting latents can then attend to one another—"}<strong>{"self-attention"}</strong>{" within that N-row workspace—using N×N comparisons."}</Prose>

<Prose>{"The next figure shows two different activities. Follow a read from the input bank into the workspace; then follow processing among the workspace vectors. The large input bank stays available. After working with the first read, updated latents can ask it another, better-informed question."}</Prose>

<PerceiverReadFigure />

<Prose>{"Thus Perceiver has two kinds of depth: processing the current latents and rereading the original input. It often chooses N much smaller than T, so most deep processing happens in the smaller array. Sharing weights across repeated blocks reduces parameter duplication while still doing another computation with updated data. These choices are described in "}<a href={"https://proceedings.mlr.press/v139/jaegle21a/jaegle21a.pdf"}>{"Perceiver, section 3.1"}</a>{". They change the computation performed; they do not promise to reproduce the output of full input-to-input attention."}</Prose>

<H3>{"5.3 Calculate two different summaries"}</H3>

<Prose>{"Use three records tagged earlier, middle and later, with numerical position keys [−1,0,1] and values [2,6,10]. Give the first read a preference for earlier positions: relative supports 4:2:1. Give the second the opposite preference, 1:2:4. Each has seven shares in total. Their weighted summaries are"}</Prose>

<div className="neural-equation"><MathBlock>{"z_1=(4\\cdot2+2\\cdot6+1\\cdot10)/7=30/7,\n\\quad\nz_2=(1\\cdot2+2\\cdot6+4\\cdot10)/7=54/7."}</MathBlock></div>

<Prose>{"The first workspace vector summarizes earlier values more strongly; the second summarizes later values more strongly. Their joint representation contains information that a single uniform mean would lose. A learned model can use richer keys and queries; this hand example isolates the read operation."}</Prose>

<Prose>{"Those supports come from ordinary attention, not a new mixing rule. A scalar query −ln 2 multiplied by keys [−1,0,1] gives scores [ln 2,0,−ln 2]. Exponentiation gives [2,1,1/2], whose normalized weights are [4/7,2/7,1/7]. Query +ln 2 reverses the preference. Query zero makes all scores equal, giving the uniform mean. The live controls below let you move continuously between these behaviors."}</Prose>

<Prose>{"A uniform query gives the mean 6 for both [2,6,10] and [0,6,12]. If a later computation receives only that single mean, it cannot distinguish those two inputs. Once the inputs have collapsed to an identical representation, any deterministic downstream function must give them the same output. This is a concrete bottleneck failure, not a claim that every one-latent nonlinear model is exactly a mean."}</Prose>

<Prose>{"Start the live workspace at its default opposite queries and inspect which end of the input each read emphasizes. Select “Use one uniform query” and compare the two displayed input fixtures. Their identical summaries expose the lost distinction. “Reset latents” restores the opposite queries; the arrays now give different representations. Adding workspace capacity is useful when it preserves information the task needs; the count alone is not a guarantee."}</Prose>

<LatentWorkspaceLab />

<H3>{"5.4 Order must travel with the record"}</H3>

<Prose>{"Suppose the path is stored in a file and you rearrange its rows. If each point keeps its original timestamp, you have changed storage order, not the path. If you give those points new timestamps, you have changed which movement happened when. A model should distinguish those edits."}</Prose>

<Prose>{"If keys and values are permuted together, cross-attention with fixed latent queries produces the same result. The score columns and their corresponding values move together, so the weighted sum is unchanged. This is desirable for a storage reordering, but it means that order must be represented if it matters to the task."}</Prose>

<Prose>{"Perceiver supplies position features alongside input features, commonly Fourier features for spatial or temporal coordinates. A frame with its correct timestamp may be moved to another storage row without changing its meaning. Assigning that frame another timestamp changes the input. In the trajectory program, each point carries a normalized position from −1 to 1."}</Prose>

<Prose>{"A Fourier feature expands a coordinate x into sine/cosine measurements at selected frequencies: for example, [x, sin(πx), cos(πx)]. At x=0 this is [0,0,1]; at x=1/2 it is [1/2,1,0]. Adding more frequencies gives the learned projections several spatial scales to combine. The original coordinate helps distinguish positions that a periodic component alone would identify. For d coordinate dimensions and K frequency pairs per coordinate, this concatenation has d(2K+1) entries. It supplies location information; it does not change the paired-permutation argument. Our small classifier uses the raw ordinal tag alone so its input representation stays transparent."}</Prose>

<PositionAttachmentFigure />

<H3>{"5.5 Ask for outputs at the places you need them"}</H3>

<Prose>{"Our path task needs one class label. Other tasks need an answer at every location—for example, the motion of each pixel between two images. The number of answers need not equal the size of the internal workspace. "}<strong>{"Perceiver IO"}</strong>{" adds an attention read on the output side: each requested output supplies a query, and the processed latent array supplies the keys and values. Original Perceiver instead aggregated its latents for tasks such as classification."}</Prose>

<Prose>{"An output query might specify a pixel coordinate, a language position or a desired modality. A coordinate alone does not determine the answer; the query asks the learned decoder to extract the relevant information from the latents. In the next diagram, count the requests, then count the returned rows. Four requests produce four answer vectors even though there are only two latent donors."}</Prose>

<Prose>{"If there are O output queries, the output has O rows. For example, four output-coordinate queries can request four motion vectors from the same two latent vectors. The output size is independent of the number of input rows and of the chosen number of latents. What can be accurately reconstructed still depends on the information those latents retained. "}<a href={"https://arxiv.org/pdf/2107.14795"}>{"Perceiver IO, sections 3.1–3.2"}</a>{"."}</Prose>

<OutputQueriesFigure />

<Prose>{"The interesting connection is that attention becomes an "}<strong>{"interface between array sizes"}</strong>{", not only a way to relate words. Input queries move information into a working representation; output queries specify what information to read from it. The later cross-attention-architectures lesson uses this perspective to connect modalities and pretrained systems."}</Prose>

<H2>{"6. Which Perceiver is allowed to generate left to right?"}</H2>

<Prose>{"The movement classifier sees a finished recording before answering. A model generating a sentence has a different job: when predicting the next word, that word must still be unknown to it. This is why we cannot simply reuse an unrestricted full-input latent array for every prediction."}</Prose>

<Prose>{"Original Perceiver uses noncausal attention and can read the whole observed input for classification. If an early language-model output consults a latent that already encoded its future target, the answer has leaked into the computation. Blocking a direct connection to the target does not help if an indirect route remains open. The mask must protect every path to the answer."}</Prose>

<Prose>{""}<strong>{"Perceiver AR"}</strong>{" aligns a smaller number of latents with selected final input positions. A latent corresponding to input position i may cross-attend only to positions j≤i; it predicts token i+1. Latent self-attention also uses the causal order. Both stages must preserve the information boundary."}</Prose>

<Prose>{"For input positions 0,1,2,3,4, take latents aligned with positions 3 and 4. The first may read inputs 0–3 and predict token 4. The second may read inputs 0–4 and predict token 5. During latent self-attention, the first latent cannot read the second: that second latent has already seen input 4, which is the first latent's target."}</Prose>

<Prose>{"In the first table below, check each latent against the inputs it is allowed to know. Then inspect the second table: the earlier latent must also be prevented from consulting the later one. Trace the displayed detour from input 4 to see why two individually plausible attention stages need coordinated masks."}</Prose>

<CausalLatentsFigure />

<Prose>{"The "}<a href={"https://icml.cc/media/icml-2022/Slides/17886.pdf"}>{"authors' ICML presentation slides, pages 3–10"}</a>{" build this input/query/target alignment incrementally. The "}<a href={"https://proceedings.mlr.press/v162/hawthorne22a/hawthorne22a.pdf"}>{"Perceiver AR paper"}</a>{" describes the architecture and training/inference choices. A latent bottleneck does not make every Perceiver variant a streaming recurrent model: retaining or rereading a long input and updating a fixed recurrent state are different execution contracts."}</Prose>

<Prose>{"Return to the document question. Transformer-XL may retain the entrance's position vector if it remains in memory, or pass its influence indirectly through later states. Griffin may preserve it in recurrent state while consulting recent text explicitly. A Perceiver read can consult it directly while the input remains available, provided the learned queries and latent capacity preserve what the task needs. Those are testable paths, not guarantees that every trained model will answer correctly."}</Prose>

<H2>{"7. Run the mechanisms, then try real movement data"}</H2>

<H3>{"7.1 The small calculations"}</H3>

<Prose>{"You have now calculated a weighted read, watched records leave a cache, separated retention from injection, and built different latent summaries. The first program turns those same small examples into reusable functions. It is the shortest route from the diagrams to an implementation you can inspect."}</Prose>

<Prose>{"Download "}<a href={"/learn-code/long-context-sequence-models-transformer-xl-griffin-perceiver/sequence_mechanisms.py"}>{"sequence_mechanisms.py"}</a>{". This complete NumPy program implements attention, a segmented memory trace, the scalar gated recurrence and the latent calculations. It also checks selected "}<strong>{"invariances"}</strong>{": changes to representation, such as jointly reordering keys and values, that should leave the result unchanged."}</Prose>

<Prose>{"Use Python 3.12 or later in your own environment:"}</Prose>

<CodeBlock language={"text"}>{"python -m venv .venv\npython -m pip install numpy\npython sequence_mechanisms.py"}</CodeBlock>

<Prose>{"Activate that environment before the "}<code>{"python -m pip"}</code>{" command, or use its Python executable explicitly. On Windows the executable is "}<code>{".venv\\Scripts\\python.exe"}</code>{"; on macOS/Linux it is "}<code>{".venv/bin/python"}</code>{". The author run used Python 3.12.14 and NumPy 2.3.5. The script writes "}<code>{"checked-results.json"}</code>{" beside itself."}</Prose>

<Prose>{"Read the following code as five actions: compare every query with the keys; exclude forbidden records; exponentiate stable scores; divide by the total support; mix the values. The "}<code>{"@"}</code>{" operator performs matrix multiplication, and "}<code>{".T"}</code>{" swaps a matrix's rows and columns. Each row stays one question throughout:"}</Prose>

<CodeBlock language={"python"}>{"scores = query @ keys.T / math.sqrt(keys.shape[1])\nscores = np.where(allowed, scores, -np.inf)\nweights = np.exp(scores - scores.max(axis=-1, keepdims=True))\nweights /= weights.sum(axis=-1, keepdims=True)\noutput = weights @ values"}</CodeBlock>

<Prose>{"Every query must have at least one legal key. Subtracting the row maximum preserves softmax while avoiding unnecessarily large exponentials. The program's reusable function also accepts a positional bias. The memory example deliberately fixes projections and omits a full neural stack so you can see exactly which input changed an output."}</Prose>

<LongContextProgram file="sequence_mechanisms.py" title="Read the complete scratch attention, cache, recurrence and latent program" />

<Prose>{"Selected executed results are:"}</Prose>

<CodeBlock language={"text"}>{"Final output with memory 0: 0.000000\nFinal output with memory 2: 3.333333\nFinal output with memory 4: 5.000000\nImpulse state after six steps, ordinary decay: 0.196608\nImpulse state after six steps, near-hold gates: 0.594668\nTwo latent outputs: 4.285714, 7.714286"}</CodeBlock>

<Prose>{"These are the hand calculations made executable: 4.285714 and 7.714286 are 30/7 and 54/7 from section 5. Start by changing one value in the latent example. The weights remain fixed because its keys and queries did not change, so the output change equals that value's change multiplied by its attention weight. This gives you a small independent check before modifying the attention function itself."}</Prose>

<H3>{"7.2 A complete trainable latent classifier"}</H3>

<Prose>{"Now ask a real question: "}<strong>{"Can a compact representation of a hand's path identify which of fifteen recorded movement classes it belongs to?"}</strong>{" The UCI Libras Movement dataset provides 360 hand trajectories extracted from videos: forty-five ordered two-dimensional positions per trajectory, followed by a class label. Its classes include circles, arcs, straight lines, waves and zigzags. These coordinates describe one hand's movement; they are not a complete representation of sign-language meaning. "}<a href={"https://archive.ics.uci.edu/dataset/181/libras%2Bmovement"}>{"UCI collection description and CC BY 4.0 license"}</a>{"."}</Prose>

<Prose>{"A point is a measured hand centroid at a selected video frame. The collectors normalized the videos into forty-five sampled positions. We therefore know the point order, but do not infer exact time intervals, hand speed in meters per second or physical distances from these unit-space coordinates. The original metadata describes four performers and two sessions; individual rows do not identify performer or session."}</Prose>

<Prose>{"To make the experiment meaningful, give each trajectory one job. "}<strong>{"Fit"}</strong>{" rows change the model's parameters. "}<strong>{"Validation"}</strong>{" rows help choose when to stop fitting. "}<strong>{"Test"}</strong>{" rows assess the resulting choice after those decisions. A copy of the same path must not appear on both sides of that assessment; otherwise apparent recognition could partly be repetition."}</Prose>

<Prose>{"Before splitting, we found thirty duplicate copies of trajectories, all with matching labels. Keeping the first occurrence of each exact coordinate sequence leaves 330 distinct trajectories. Within each class, a fixed NumPy seed of 73 determines a permutation. Reserve the last four trajectories for test; use the first two-thirds, rounded down, for fitting and the rest for validation. This gives "}<strong>{"220 fit, 50 validation and 60 test trajectories"}</strong>{", with no exact coordinate sequence crossing roles. Because rows lack performer/session identifiers, this experiment cannot establish performance on a new performer or recording session."}</Prose>

<Prose>{"These forty-five-point sequences are a practical CPU exercise in latent sequence processing. They let us inspect learned reads and compare ways of preserving shape. They do not measure long-context language-model performance."}</Prose>

<Prose>{"Download "}<a href={"/learn-code/long-context-sequence-models-transformer-xl-griffin-perceiver/movement_libras.data"}>{"movement_libras.data"}</a>{" and "}<a href={"/learn-code/long-context-sequence-models-transformer-xl-griffin-perceiver/latent_trajectory_classifier.py"}>{"latent_trajectory_classifier.py"}</a>{" into one directory. The "}<a href={"/learn-code/long-context-sequence-models-transformer-xl-griffin-perceiver/data-provenance.md"}>{"data record"}</a>{" supplies attribution, source-row roles, duplicate handling and exact transformations. The program runs offline once dependencies are installed:"}</Prose>

<CodeBlock language={"text"}>{"python -m pip install numpy torch scikit-learn\npython latent_trajectory_classifier.py"}</CodeBlock>

<Prose>{"The tested snapshot is PyTorch 2.14.0+cpu, NumPy 2.3.5 and scikit-learn 1.9.1. The program uses two CPU threads and writes "}<code>{"trajectory-results.json"}</code>{" and "}<code>{"small-fits.npz"}</code>{". The latter stores the small learned arrays and selected read weights."}</Prose>

<Prose>{""}<strong>{"First give simpler methods a fair attempt."}</strong>{" A baseline is a comparison that tells us what the new machinery adds. Transform each coordinate using the fixed rule 2x−1, then average the forty-five x coordinates and forty-five y coordinates. A logistic classifier receives just those two center coordinates. This is the same information loss shown by the straight and zigzag paths in section 5."}</Prose>

<Prose>{"A second logistic classifier receives all ninety coordinates in their original order. It can distinguish shapes that share a center, although its decision function is linear in the supplied coordinates. Both baselines use C=1, fit only the fit rows and use no learned preprocessing across roles. The "}<a href={"/learn/path/full-curriculum/linear-logistic-regression?module=classical-ml"}>{"linear and logistic regression lesson"}</a>{" develops these classifiers; here the comparison asks whether preserving shape matters more than adding a complicated model."}</Prose>

<Prose>{""}<strong>{"Then learn which information to gather."}</strong>{" The latent model appends an ordinal tag running from −1 to 1 to each coordinate pair. Each of the forty-five rows now has three numbers: x, y and order. A learned projection turns each row into 24 features. N learned latent vectors read those features, interact with each other and pass through a feed-forward block. The model repeats that read/process step once with shared weights, averages the final latents and produces fifteen "}<strong>{"logits"}</strong>{"—unrestricted numerical scores, one for each class. Softmax converts them into class probabilities. We compare N=1 and N=4 under the same protocol."}</Prose>

<Prose>{"Read the diagram from measured geometry to model features. The ordinal tag tells the model where a point belongs in the movement; it is neither another spatial coordinate nor a measurement of elapsed seconds. Count rows at each transition: forty-five input rows can feed one or four latents while the answer still has fifteen class scores."}</Prose>

<TrajectoryArchitectureFigure />

<Prose>{"The central loop follows that diagram. "}<code>{"cross"}</code>{" reads input records; "}<code>{"self_attention"}</code>{" lets the working vectors exchange information; "}<code>{"feedforward"}</code>{" transforms each working vector's features. Each "}<code>{"latent + update"}</code>{" is a residual update, preserving a path for the current representation while adding new information:"}</Prose>

<CodeBlock language={"python"}>{"inputs = self.input_projection(frames)\nlatent = self.latent.unsqueeze(0).expand(len(frames), -1, -1)\nfor _ in range(2):\n    update, weights = self.cross(self.cross_norm(latent), inputs, valid)\n    latent = latent + update\n    normalized = self.self_norm(latent)\n    update, _ = self.self_attention(normalized, normalized)\n    latent = latent + update\n    latent = latent + self.feedforward(self.final_norm(latent))\nlogits = self.classifier(latent.mean(dim=1))"}</CodeBlock>

<Prose>{"The complete file defines the projections, attention, normalization, feed-forward block, duplicate handling, data roles, training, metrics and saving. First trace "}<code>{"inputs"}</code>{": it stays available across both rounds. Then trace "}<code>{"latent"}</code>{": it changes after every update. The second round reuses the same learned projection parameters, but its updated queries generally produce different attention weights over the input records. This is a small Perceiver-style classifier with inspectable dimensions, rather than a reproduction of the research model's scale."}</Prose>

<LongContextProgram file="latent_trajectory_classifier.py" title="Read the complete trainable classifier, baselines and evaluation program" />

<Prose>{""}<strong>{"How do the starting queries learn what to ask?"}</strong>{" Training penalizes assigning low probability to the recorded class. For a trajectory with true class c, cross-entropy is −ln p(c). Its derivative with respect to logit k is pₖ−1[k=c], where the indicator is 1 for the correct class and 0 otherwise. At uniform predictions, the correct logit's derivative is −14/15 and each other derivative is +1/15. Subtracting a small multiple of this gradient raises the correct score relative to the others."}</Prose>

<Prose>{"Backpropagation carries that signal through the classifier, latent processing and attention reads to the starting queries and projections. No separate labels say where each latent should look. The classification objective supplies the training signal. Pass logits, not precomputed probabilities, into PyTorch's "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html"}>{"CrossEntropyLoss"}</a>{", which performs the required stable log-softmax internally."}</Prose>

<Prose>{"Train for eighty full-batch epochs with Adam at learning rate .005. Each run retains the epoch with lowest validation cross-entropy. Latent counts 1 and 4, seeds 11 and 29 and the training settings are declared before interpreting test results. Evaluate each selected checkpoint on test only after its validation-based selection."}</Prose>

<NeuralTable caption={"7.2 A complete trainable latent classifier"} headers={[<>{"Model"}</>,<>{"Seed"}</>,<>{"Stored parameters"}</>,<>{"Selected epoch"}</>,<>{"Fit errors /220"}</>,<>{"Validation errors /50"}</>,<>{"Test errors /60"}</>]} rows={[[<>{"Mean coordinates + logistic"}</>,<>{"—"}</>,<>{"45"}</>,<>{"—"}</>,<>{"183"}</>,<>{"45"}</>,<>{"54"}</>],[<>{"Ordered coordinates + logistic"}</>,<>{"—"}</>,<>{"1,365"}</>,<>{"—"}</>,<>{"35"}</>,<>{"17"}</>,<>{"22"}</>],[<>{"One latent"}</>,<>{"11"}</>,<>{"7,815"}</>,<>{"80"}</>,<>{"95"}</>,<>{"36"}</>,<>{"32"}</>],[<>{"One latent"}</>,<>{"29"}</>,<>{"7,815"}</>,<>{"68"}</>,<>{"67"}</>,<>{"24"}</>,<>{"25"}</>],[<>{"Four latents"}</>,<>{"11"}</>,<>{"7,887"}</>,<>{"75"}</>,<>{"68"}</>,<>{"29"}</>,<>{"28"}</>],[<>{"Four latents"}</>,<>{"29"}</>,<>{"7,887"}</>,<>{"72"}</>,<>{"90"}</>,<>{"30"}</>,<>{"34"}</>]]} />

<Prose>{"The mean baseline makes 54/60 errors: knowing a path's center is a poor substitute for its shape. The ordered linear model makes 22/60 errors, about 36.7%, and outperforms all four small latent runs here. The latent models improve on the mean baseline, but their 25–34 test errors show that having learnable selective reads is not enough to make fitting reliable on this small sample. Four latents do not consistently beat one."}</Prose>

<Prose>{"This comparison is useful precisely because it separates representational possibility from achieved performance. The models are not parameter-matched; the simple ordered baseline has a helpful representation supplied directly. Two seeds show sensitivity in these runs, not a confidence interval or a universal architecture ranking. No setting was changed after seeing these outcomes."}</Prose>

<TrajectoryEvidenceFigure />

<Prose>{"The error plot makes two comparisons visible: changing the representation from center to ordered coordinates, and changing the latent model's seed or capacity. Read lower marks as fewer mistakes, using the printed denominators to distinguish validation from test. The disagreement among runs is part of the result."}</Prose>

<Prose>{"The live workbench below uses a "}<strong>{"frozen"}</strong>{" fitted model: changing a point recomputes its output immediately, but does not retrain it. Begin with a full validation path and inspect its shape. Select a point, move it, and compare the path, class probabilities and selected latent's read weights. A large read weight means the point contributes strongly to that particular read; it is not proof that the point caused the final class decision on its own."}</Prose>

<LongContextTrajectoryLab />

<Prose>{"Look for two outcomes: an edit that changes the winning class, and a smaller edit that changes probabilities while the winner stays the same. The latter is still a real response; the winning label hides all changes short of crossing a decision boundary. Compare the mean-coordinate baseline on the same retained points to see whether moving the center alone helps explain the response. An edited hypothetical path has no automatically known new class label, so a changed prediction is evidence of model sensitivity, not proof of correct recognition."}</Prose>

<Prose>{"Two null checks passed for all four fitted models. Permuting complete coordinate-plus-position records preserves logits within floating-point tolerance. Appending five points filled with 1,000 but masked out also preserves logits. If masked points alter the output, a mask or an earlier operation has admitted information that was supposed to be absent."}</Prose>

<H3>{"7.3 Move from explicit operations to maintained library operators"}</H3>

<Prose>{"Once the small calculations are clear, a maintained operator can provide the same operation more efficiently. The useful question is whether it receives the same tensors, masks and state—not whether its API name resembles the architecture. The bridge program checks matching inputs, outputs and gradients before you substitute operators in a larger system."}</Prose>

<Prose>{"Keep the three files together. "}<a href={"/learn-code/long-context-sequence-models-transformer-xl-griffin-perceiver/sequence_mechanisms.py"}>{"sequence_mechanisms.py"}</a>{" implements transparent arithmetic. "}<a href={"/learn-code/long-context-sequence-models-transformer-xl-griffin-perceiver/latent_trajectory_classifier.py"}>{"latent_trajectory_classifier.py"}</a>{" adds trainable projections, a learning loop and evaluation. "}<a href={"/learn-code/long-context-sequence-models-transformer-xl-griffin-perceiver/memory_library_bridge.py"}>{"memory_library_bridge.py"}</a>{" reuses the classifier's "}<code>{"Attention"}</code>{" class and compares its operations with library counterparts. This deliberate import avoids maintaining a second subtly different attention implementation."}</Prose>

<Prose>{""}<strong>{"Segment memory:"}</strong>{" "}<code>{"check_segmented_attention"}</code>{" builds a real K/V cache, appends the current segment, constructs a causal mask using absolute retained positions, then keeps the last four entries. Its explicit stable score→softmax→weighted-sum path is compared to "}<code>{"F.scaled_dot_product_attention"}</code>{" on exactly the same projected values, including a final short segment. In SDPA's Boolean mask, "}<code>{"True"}</code>{" means allowed. Calling "}<code>{"is_causal=True"}</code>{" on a nonsquare query/cache matrix without considering alignment can select the wrong history. Detaching cache tensors cuts gradient history; it does not reset their values or make a stream's cache suitable for another stream."}</Prose>

<Prose>{"This is the segment-memory mechanism described in §3. It is not a complete Transformer-XL checkpoint: the paper also specifies layerwise hidden-state recurrence, relative-position scoring, normalization and a language-model head. Our illustrative distance bias and already-projected one-layer K/V cache must keep those labels. The later "}<a href={"/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals"}>{"Self-Attention lesson"}</a>{" owns the full projected multihead construction. It is a deeper follow-on reference; the local small read supplies the operations needed here regardless of that lesson's implementation status."}</Prose>

<Prose>{""}<strong>{"Recurrent memory:"}</strong>{" "}<code>{"rglru_reference"}</code>{" goes beyond the earlier fixed scalar gates. It calculates learned block-diagonal input and retention gates, stable "}<code>{"-expm1(2*log_decay)"}</code>{" injection scale, per-example reset positions and each updated vector state. No recurrent-cell API hides that computation. The optional "}<code>{"check_griffin_component"}</code>{" copies actual DeepMind "}<code>{"RGLRU"}</code>{" parameters, compares outputs and the last cache, continues in two chunks and checks gradients. The native implementation clips an extreme square-root derivative for training stability; the comparison explicitly uses moderate float32 decay outside that clipped regime. This is the RG-LRU component, not an entire Griffin language model or its short temporal convolution/local-attention schedule. The "}<a href={"https://raw.githubusercontent.com/google-deepmind/recurrentgemma/main/recurrentgemma/torch/layers.py"}>{"official recurrent component source"}</a>{", inspected 22 September 2026, owns the exact package convention."}</Prose>

<Prose>{""}<strong>{"Latent workspace:"}</strong>{" "}<code>{"check_latent_read"}</code>{" reuses our existing learned "}<code>{"Attention"}</code>{" class, obtains its projected query/key/value tensors and passes those exact tensors to SDPA. It then applies the same output projection and compares outputs plus gradients for source, latent query and every parameter. The complete classifier alternates this read with latent self-attention and its feed-forward block. A generic full-input model would erase the architectural question; replacing only the matching read preserves it."}</Prose>

<Prose>{"Run "}<code>{"python memory_library_bridge.py"}</code>{" with PyTorch, NumPy and scikit-learn. Adding "}<code>{"--griffin"}</code>{" requires a compatible "}<code>{"recurrentgemma"}</code>{" installation with its Torch dependencies. The ordinary Torch comparisons are executed in the implementation checks. The optional specialist program is supplied as executable instructional content; optional package parity is not claimed executed. Record the resolved package version when running it."}</Prose>

<LongContextProgram file="memory_library_bridge.py" title="Read the matched library operators, RG-LRU reference and gradient comparisons" />

<Prose>{"The explicit attention reference materializes query×visible-key scores; the normal SDPA backend can avoid retaining that matrix under supported conditions. A bounded segment of length Q with M retained positions uses O(Q(Q+M)) score cells per head in the reference. RG-LRU streaming retains one width-sized state per example, while training retains its needed history. Latent reads use L×S score cells for L latents and S source positions. These are separate computational contracts, so there is no one “linear-memory” claim covering all three."}</Prose>

<Prose>{""}<strong>{"Change the stream."}</strong>{" Set segment length 4, keep only 2 previous positions, and process 11 inputs. Then place a recurrent reset at position 6 while preserving every input. Compare the supplied reference and package paths again."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"The last segment has 3 queries. Cache truncation discards earlier keys; a recurrent reset clears prior state for exactly the selected example."}</Prose>

</details>

<details>

<summary>Solution and success criteria</summary>

<Prose>{"Build legal key positions from the retained absolute indices plus the current segment, mask future indices, and trim the cache only after obtaining that segment's outputs. The first query of the last segment sees indices 6, 7, 8; its later queries can also see 9 and 10 as they arrive. In RG-LRU, set "}<code>{"segment_pos[:,6]=0"}</code>{" and restart its within-document counter; the reset input uses the special fresh-document injection and cannot depend on the earlier cache. Verify unaffected earlier outputs and a changed post-reset state. This tests state ownership rather than memorizing model names."}</Prose>

</details>

<H2>{"8. Deeper: gradients, budgets and a useful evaluation plan"}</H2>

<Prose>{"This section is a deeper route. It makes the architectures' costs and training boundaries precise after their forward mechanisms are familiar."}</Prose>

<H3>{"8.1 Forward memory and gradient memory"}</H3>

<Prose>{"A detached cache illustrates two different graphs: the graph of values used in a prediction, and the graph followed by differentiation. Transformer-XL keeps an old representation in the value graph while cutting its backward connection to its earlier computation. Replacing the cache by zero changes the forward computation; detaching it does not change its forward numbers."}</Prose>

<Prose>{"For a scalar example, let old state m=2w and current prediction y=w·stopgrad(m). At w=3, m=6 and y=18. The detached derivative with respect to the current w is 6. If we differentiate through the original m=2w as well, y=2w² has derivative 12. This difference explains truncated credit assignment. It does not imply the old state was ignored."}</Prose>

<DetachedMemoryFigure />

<Prose>{"The same distinction matters for recurrent training. Constant inference state does not mean backpropagation stores a constant number of intermediate activations. Training may retain a sequence, recompute activations, use checkpointing or use a specialized scan. State-space and sequence-parallel lessons develop those choices."}</Prose>

<H3>{"8.2 Count the particular resource you mean"}</H3>

<Prose>{"The smaller workspace is motivated by a concrete count: how many question–record pairs must be compared? Two queries reading five records make ten score cells. Applying attention to every pair in a 4,096-position input makes over sixteen million. This count explains why changing the array sizes can matter before considering hardware or implementation speed."}</Prose>

<Prose>{"Let T be input length, L segment length, M retained memory, W local window, N latent count and D the number of latent layers. Hold channel widths and head counts fixed for the following attention-interaction counts:"}</Prose>

<NeuralTable caption={"8.2 Count the particular resource you mean"} headers={[<>{"Computation"}</>,<>{"Leading interaction count"}</>,<>{"What remains separately important"}</>]} rows={[[<>{"Dense attention on T positions"}</>,<>{"O(T²) per layer"}</>,<>{"Projection/MLP work, causal mask, training activations"}</>],[<>{"Segments of length L with memory M"}</>,<>{"O(T(L+M)) per layer"}</>,<>{"How memory is filled and detached; hidden-state storage"}</>],[<>{"Local attention with window W"}</>,<>{"O(TW) per layer"}</>,<>{"Recurrent blocks and their state in a hybrid"}</>],[<>{"One Perceiver input read plus D latent layers"}</>,<>{"O(TN+DN²)"}</>,<>{"Additional input reads, channel projections, output decoding"}</>],[<>{"Perceiver IO with O output queries"}</>,<>{"Add O(ON)"}</>,<>{"Output features and output-head work"}</>]]} />

<Prose>{"If there are R separate input reads, count R·TN, not just TN. If N grows proportionally to T, the bottleneck no longer gives linear input scaling under fixed depth. The recurrent scalar/vector update is linear in sequence length for fixed state width; obtaining its input projections is additional work."}</Prose>

<Prose>{"For T=4,096, a dense T×T score array contains 16,777,216 cells. With L=256 and M=128, the rectangular per-segment upper count is 16·256·384=1,572,864 cells across the sequence. Window W=128 gives at most 524,288 score slots. One read to N=32 latents plus eight latent layers gives 4,096·32+8·32²=139,264 cells. These are algebraic score counts under the stated model, not GPU timings or exact complete-model FLOPs."}</Prose>

<MemoryBudgetsFigure />

<Prose>{"Attention computation and persistent key/value cache size have different growth. A full multi-head KV cache for one layer, T positions, total key/value width d and b bytes per stored element uses 2Tdb bytes. With T=4,096, d=64 and b=2, that is 1,048,576 bytes, or 1 MiB. Restricting the retained window to 128 positions gives 32,768 bytes, or 32 KiB. The cache grows linearly with T; the straightforward attention score matrix has quadratic cells. Fused attention can avoid materializing that entire matrix without changing which pairs the operator compares."}</Prose>

<Prose>{"Transformer-XL may cache hidden activations rather than the exact projected KV representation assumed in that formula. Multi-query/grouped-query models change stored KV width. Use each implementation's actual cache contract rather than applying the MHA formula to every architecture with “attention” in its name."}</Prose>

<H3>{"8.3 Linear recurrences can have parallel training algorithms"}</H3>

<Prose>{"An update can be understood as an instruction such as “multiply the current state by .5, then add 2.” Follow it with “multiply by .8, then add 1.” Together they mean “multiply by .4, then add 2.6.” Combining the instructions first gives the same result for every starting state. That ability to combine updates is the entry point to parallel recurrence algorithms."}</Prose>

<Prose>{"For input-dependent coefficients already computed from the input, a recurrent step is an affine map h↦a⊙h+b. Two steps compose to"}</Prose>

<div className="neural-equation"><MathBlock>{"(a_2,b_2)\\circ(a_1,b_1)\n=(a_2\\odot a_1,\\ a_2\\odot b_1+b_2)."}</MathBlock></div>

<Prose>{"Composition is associative. Therefore a parallel prefix algorithm can combine these maps in a tree rather than evaluating every dependency with a serial host-language loop. This preserves the mathematical recurrence, with ordinary floating-point-order differences. It is different from making the gates depend on the previous hidden state, which would prevent precomputing those affine maps in the same way. The next state-space lesson will use this distinction repeatedly."}</Prose>

<AffineScanFigure />

<H3>{"8.4 Evaluate the information claim, not just input acceptance"}</H3>

<Prose>{"For the document task, construct examples with the queried fact at several distances, distractors with similar wording and multiple independent facts. Hold the question and answer format fixed. Compare short recent context, the intended long-context method, and an explicit retrieval baseline that supplies the relevant passage. Track accuracy by distance and number of competing facts, together with actual retained memory and measured latency if performance is being studied."}</Prose>

<Prose>{"A repeated fact at the end is a useful control: it removes the need to remember its early occurrence. An unrelated prefix is another control: extra tokens should not help a task whose answer is entirely local. For summarization, use a different assessment of factual coverage and attribution; success on one isolated “needle” is not a complete test of document understanding."}</Prose>

<Prose>{"For the movement exercise, the held-out unit is a complete trajectory, not a shuffled point. Splitting points would mix parts of the same path across fit and test. Exact duplicate trajectories also belong to one role or must be removed before splitting. A new-performer or new-session claim needs the corresponding identifiers and a grouped protocol, which these public rows do not provide. These decisions determine what “works” means before fitting."}</Prose>

<Prose>{"The named papers report historical experiments under their own data and hardware settings. We use them to investigate mechanisms and research evidence, without transplanting their numerical speedups into the small CPU examples here. The important practical habit is to preserve one interpretable change, a credible baseline, the task's information boundary and a result you can inspect."}</Prose>

<section className="lesson-ending lesson-ending--practice" data-lesson-ending="practice"><H2>{"9. Practice: implement a change and explain its effect"}</H2>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"1. Repair a masked weighted average"}</H3>

<Prose>{"A one-dimensional query gives unnormalized weights [3,1,5] and values [7,3,100]. The last position is in the future. Calculate the legal output and explain why dividing by 9 is wrong."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"The future item disappears from both the weighted numerator and the normalizing denominator."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"The result is (3·7+1·3)/(3+1)=6. Dividing by 9 would retain the future score in the denominator, shrinking the legal information. A mask is a restriction on participating key/value positions, not only a zeroing of future values."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"2. Build a cache counterexample"}</H3>

<Prose>{"Process [A,B,C,D,E,F] in segments of length 3 with memory length 2. At the query for E, which earlier/current positions are directly legal? Put the only informative value at a position that cannot be consulted. What memory length would recover it?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"The second segment starts with the retained tail of the first. Current-segment future positions remain masked."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"The second segment retains B and C, then contains D,E,F. The query at E can use B,C,D,E; F is future, and A was evicted. An informative A is a counterexample to direct retrieval with M=2. M=3 would retain A,B,C. In a deeper contextual model an indirect influence could survive elsewhere, but this question concerns direct access in the stated layer."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"3. Separate the two gates"}</H3>

<Prose>{"An existing scalar state is 2. For the next three steps there is no incoming signal, and effective decay is .75. Calculate the final state. Would reducing only the input gate preserve it? What effective decay would preserve it exactly in this mathematical fixture?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"There is no incoming term to suppress. Follow the multiplier applied to the old state."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"The final state is 2·.75³=.84375. Changing the input gate does nothing when the gated input is already zero. Effective decay 1 preserves state exactly. In RG-LRU that is the limiting recurrence-gate setting r=0 for a fixed base in (0,1); finite sigmoid logits approach the limit."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"4. Diagnose a latent collision"}</H3>

<Prose>{"A model replaces three scalar values by their uniform mean and then applies an arbitrary deterministic classifier. Construct two different inputs with the same mean but different last values. Can retraining only the downstream classifier let it identify the last value perfectly on both?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Make the sum agree while changing the last element. The classifier receives only one number."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"[1,4,7] and [0,4,8] both produce mean 4, but their last values differ. Any deterministic classifier of that mean receives identical input in both cases and returns the same result. It needs a richer representation, another input read or a changed task. This proof is about the explicitly uniform mean; a learned one-latent attention mechanism need not compute that mean."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"5. Find the indirect causal leak"}</H3>

<Prose>{"Latent z₂ predicts token 3 and may read inputs through position 2. Latent z₃ predicts token 4 and may read inputs through position 3. Both cross-attention masks are correct, but z₂ can attend to z₃ in the next layer. Explain the leak and repair it."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Trace the target of z₂ through the other latent, rather than inspecting only its direct input edges."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"Input token 3 can enter z₃ and then reach z₂. That reveals z₂'s target through a two-step path. The latent self-attention mask must also respect aligned position: z₂ cannot read z₃. End-to-end causality is a property of all paths through the computation."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"6. Compare two memory budgets"}</H3>

<Prose>{"A full multi-head KV cache has 12 layers, 2,048 positions, 8 KV heads, head width 32 and float16 storage. Calculate its size in MiB for one sequence. Then use a retained window of 256 with everything else unchanged. Name one allocation excluded from this count."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Count keys and values separately; a MiB is 2²⁰ bytes."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"2·12·2,048·8·32·2=25,165,824 bytes=24 MiB. Replacing 2,048 by 256 gives 3 MiB. Parameters, activation storage, allocator overhead and other state are examples of excluded allocations. This formula assumes the specified 8 stored KV heads, not a multi-query cache with shared keys and values."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"7. Design a meaningful trajectory ablation"}</H3>

<Prose>{"Keep the supplied data roles. Compare the complete-coordinate/position permutation with a transformation that reverses the coordinate sequence but reassigns increasing positions. Predict which transformation should preserve the frozen latent model's output, then perform the comparison on one validation trajectory. Explain what a difference would establish."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"The first transformation reorders records. The second changes which measured frame occupies each temporal position."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"The paired permutation preserves the weighted reads and therefore the logits up to floating-point roundoff. Reassigning positions changes model inputs, so outputs may differ; they need not differ for every trajectory or fitted model. Retain the original and transformed logits, valid-point count and original label. A difference establishes sensitivity to that temporal reassignment in this example, not that temporal order universally improves movement classification. Never select a new model using the held-out test trajectories to make the result look stronger. The edited path's original label is a reference, not certified ground truth for the hypothetical edit."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"8. A detached value still matters"}</H3>

<Prose>{"Let m=4w and y=2w·stopgrad(m). At w=2 calculate y, the detached derivative with respect to w, and the derivative if m is not detached. Explain the distinction in words."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Hold the numerical memory value fixed in the first derivative; substitute m=4w before the second."}</Prose>

</details>

<details>

<summary>Solution</summary>

<Prose>{"m=8 and y=32. With the cached value fixed, dy/dw=2m=16. Differentiating through m gives y=8w² and dy/dw=16w=32. Both computations use the same forward memory value, but one omits the earlier operation's contribution to the gradient."}</Prose>

</details>

<Prose>{"You are ready to continue when you can draw the legal information paths, calculate one attention read and recurrent update, explain a bottleneck collision, and distinguish a stored-state count from a measured performance claim. The next topic is "}<a href={"/learn/path/full-curriculum/state-space-models-s4-mamba-mamba-2?module=deep-learning-fundamentals"}>{"State Space Models (S4, Mamba, Mamba-2)"}</a>{". It develops how a hidden state evolves, when a recurrence becomes a convolution, and why input-dependent selection changes the computation."}</Prose></div></section>

<section className="lesson-ending lesson-ending--resources" data-lesson-ending="resources"><H2>{"10. References & another way to learn it"}</H2>

<Prose>{"For a second explanation, choose the source that matches the mechanism you are trying to picture. The creator articles below offer a gentler visual route; the papers give the exact architecture. Their reported benchmarks belong to their publication settings, not a current universal ranking."}</Prose>

<ul><li>{""}<a href={"https://research.google/blog/transformer-xl-unleashing-the-potential-of-attention-models/"}>{"Google Research, Transformer-XL: Unleashing the Potential of Attention Models"}</a>{", creator article. Start with its context-fragmentation example and segment-recurrence illustrations when it is unclear why a chunk boundary loses information. Then return to our five-record cache trace."}</li><li>{""}<a href={"https://arxiv.org/pdf/1901.02860"}>{"Dai and colleagues, Transformer-XL"}</a>{", research paper. Read section 3 with the segment/layer drawing beside you; appendix B is an advanced route to efficient relative-position score construction. The reported model comparisons are from the paper's 2019 setting."}</li><li>{""}<a href={"https://arxiv.org/pdf/2402.19427"}>{"De and colleagues, Griffin"}</a>{", research paper. Section 2 gives the full blocks and RG-LRU equations; appendix A discusses the gate's behavior and stable parameterization. Useful after the scalar recurrence exercise."}</li><li>{""}<a href={"https://developers.googleblog.com/en/gemma-explained-recurrentgemma-architecture/"}>{"Google Developers, RecurrentGemma architecture"}</a>{", illustrated article. A practical second tour through projections, recurrent layers and local attention. Use the paper for exact mathematical conventions and treat model-specific dimensions as the article's versioned examples."}</li><li>{""}<a href={"https://proceedings.mlr.press/v139/jaegle21a/jaegle21a.pdf"}>{"Jaegle and colleagues, Perceiver"}</a>{", canonical research paper. Section 3 explains the asymmetric read, latent processing, iterative reads and position features. Start here for the bottleneck architecture rather than an unverified implementation summary."}</li><li>{""}<a href={"https://huggingface.co/blog/perceiver"}>{"Hugging Face, Perceiver IO: a scalable, fully-attentional model that works on any modality"}</a>{", implementation-author tutorial. Its architecture illustrations and shape walkthrough help you track which array supplies queries and which supplies keys and values. The linked examples are another implementation route; they are not the program executed for this lesson's results."}</li><li>{""}<a href={"https://arxiv.org/pdf/2107.14795"}>{"Jaegle and colleagues, Perceiver IO"}</a>{", research paper. Sections 3.1–3.2 explain output queries; the optical-flow example supplies a concrete setting where the output is an array rather than one class."}</li><li>{""}<a href={"https://deepmind.google/blog/building-architectures-that-can-handle-the-worlds-data/"}>{"DeepMind, Building architectures that can handle the world's data"}</a>{", creator article with architecture illustrations and linked audiovisual demonstrations. A less algebraic route to input/latent/output roles; the demonstrations are examples from the authors' historical systems."}</li><li>{""}<a href={"https://proceedings.mlr.press/v162/hawthorne22a/hawthorne22a.pdf"}>{"Hawthorne and colleagues, Perceiver AR"}</a>{", research paper, and "}<a href={"https://icml.cc/virtual/2022/spotlight/17886"}>{"ICML 2022 spotlight with slides and a listed video"}</a>{". The inspected "}<a href={"https://icml.cc/media/icml-2022/Slides/17886.pdf"}>{"slide deck"}</a>{" builds the two causal masks step by step. The conference page lists a video; playback access was not verified for this packet. Read the slides independently of it."}</li><li>{""}<a href={"https://deepmind.google/blog/perceiver-ar-general-purpose-long-context-autoregressive-generation/"}>{"DeepMind, Perceiver AR: general-purpose, long-context autoregressive generation"}</a>{", creator explanation. Its illustrated progression from input positions to aligned latents offers another route through why a generative Perceiver needs causal information paths."}</li><li>{""}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html"}>{"PyTorch attention documentation"}</a>{", API reference for extending the explicit teaching implementation. Inspect mask semantics and dropout behavior when moving between APIs; a Boolean mask does not have the same meaning in every attention interface."}</li><li>{""}<a href={"https://archive.ics.uci.edu/dataset/181/libras%2Bmovement"}>{"UCI Libras Movement"}</a>{", original data documentation. Read the sampling, coordinate layout and recording context before modifying the real example. Attribution and the exact retained data transformation are in the supplied data record."}</li></ul></section>
  </div>,
};
