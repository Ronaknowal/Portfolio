// Generated from the complete prepared manuscript, preserving all twelve sections and changed practice.
import {Prose,H2,H3,CodeBlock} from '../../components/content';
import {Math as InlineMath,MathBlock} from '../../components/content/Math.jsx';
import {NeuralTable} from '../../components/lesson-labs/NeuralLessonElements.jsx';
import {HybridMemoryLab,HybridBudgetLab,HybridRouterLab} from '../../components/lesson-labs/HybridJambaLabs.jsx';
import {JambaOpeningFigure,JambaWorkedReadFigure,JambaCollisionFigure,JambaResidualFigure,JambaDepthFigure,JambaStateFigure,JambaHeadsFigure,JambaMemoryFigure,JambaArithmeticFigure,JambaRouterWorked,JambaParameterFigure,JambaReleaseFigure,JambaCandidatesFigure,JambaRequestsFigure,JambaAlternativesFigure} from '../../components/lesson-labs/HybridJambaDiagrams.jsx';
import {JambaSelectivePipeline,JambaScanFigure,JambaPositionFigure,JambaPrecisionFigure,JambaRewindFigure,JambaSharedVariantsFigure} from '../../components/lesson-labs/HybridJambaMechanisms.jsx';
import {JambaWorkedStrokeFigure,JambaTrainingFigure,JambaContinuationFigure,HybridStrokeLab,JambaProgram,JambaDownloads} from '../../components/lesson-labs/HybridJambaStudy.jsx';
export default {title:'Hybrid SSM–Transformer Architectures: Jamba and Complementary Memory',readTime:'~90 min read + investigations and practice',content:()=> <div className="neural-lesson neural-lesson-neutral jamba-lesson">
<Prose opening="exploration">{""}<strong>{"Explore as you read."}</strong>{" Relabel a record and compare a running summary with a query-based read. Then change a memory budget, follow an expert route, and edit a real pen trajectory to see how its saved model uses recurrent state and attention history."}</Prose>

<Prose>{"A document assistant may need to follow a developing argument and then retrieve the exact amount beside an invoice number. A handwriting recognizer may need to follow a pen's movement and then compare the closing stroke with an earlier turn. These are related jobs, but the most convenient memory for one is not automatically the most convenient memory for the other."}</Prose>

<Prose>{"A "}<strong>{"hybrid sequence model"}</strong>{" uses more than one kind of sequence-processing operation inside a single network. In the family studied here, some layers update a compact recurrent state; others compare a query with stored keys and read the corresponding values. Jamba combines Mamba state-space layers, attention layers and, at selected feed-forward positions, sparsely chosen experts."}</Prose>

<Prose>{"The previous lesson, "}<a href={"/learn/path/full-curriculum/neural-ode-continuous-depth-models?module=deep-learning-fundamentals"}>{"Neural ODEs"}</a>{", asked what happens when computation across depth becomes a continuous trajectory. Here depth is a discrete sequence of different operations. A second axis, the positions in the input, must remain separate: a model can have 32 layers while reading 262,144 tokens."}</Prose>

<Prose>{"You will learn to trace those two axes, calculate the memory a request actually carries, explain the difference between sparse expert selection and sparse attention, and investigate a small trained hybrid without mistaking it for a miniature benchmark of a giant language model."}</Prose>

<Prose opening="route">{""}<strong>{"Core route:"}</strong>{" follow §§1–8, then solve the first six practice questions. The optional branches on expert arithmetic, cross-layer sharing and deployment experiments add depth without being prerequisites for understanding the central mechanism."}</Prose>

<H2>{"1. Two useful questions need two different reads"}</H2>

<Prose>{"Imagine three labelled measurements arriving in order:"}</Prose>

<NeuralTable caption={"1. Two useful questions need two different reads"} headers={[<>{"Position"}</>,<>{"Label"}</>,<>{"Measurement"}</>]} rows={[[<>{"1"}</>,<>{"A"}</>,<>{"2"}</>],[<>{"2"}</>,<>{"B"}</>,<>{"7"}</>],[<>{"3"}</>,<>{"C"}</>,<>{"4"}</>]]} />

<Prose>{"One question is “What level has the stream recently been around?” Another is “What measurement belongs to A?” The first invites a running summary. The second invites a label-sensitive lookup."}</Prose>

<JambaOpeningFigure/>

<Prose>{"For a simple recency-weighted average, keep a numerator "}<InlineMath>{"s_t"}</InlineMath>{" and a normalizer "}<InlineMath>{"n_t"}</InlineMath>{":"}</Prose>

<div className="neural-equation"><MathBlock>{"s_t=\\lambda s_{t-1}+v_t,\\qquad\nn_t=\\lambda n_{t-1}+1,\\qquad\nr_t=s_t/n_t."}</MathBlock></div>

<Prose>{"Start with "}<InlineMath>{"s_0=n_0=0"}</InlineMath>{". With "}<InlineMath>{"\\lambda=1/2"}</InlineMath>{", the state evolves as follows:"}</Prose>

<NeuralTable caption={"1. Two useful questions need two different reads"} headers={[<>{"After record"}</>,<>{""}<InlineMath>{"s_t"}</InlineMath>{""}</>,<>{""}<InlineMath>{"n_t"}</InlineMath>{""}</>,<>{"Recency-weighted average"}</>]} rows={[[<>{"A:2"}</>,<>{"2"}</>,<>{"1"}</>,<>{"2"}</>],[<>{"B:7"}</>,<>{"8"}</>,<>{"1.5"}</>,<>{"5.333333"}</>],[<>{"C:4"}</>,<>{"8"}</>,<>{"1.75"}</>,<>{"4.571429"}</>]]} />

<Prose>{"The final answer gives the last observation four times the weight of the first. Only two numbers remain in memory, regardless of how many records arrived. This particular summary does not even use the labels."}</Prose>

<Prose>{"For the lookup, assign a compatibility score "}<InlineMath>{"\\log 9"}</InlineMath>{" to a matching label and zero to each nonmatch. Softmax converts these scores into weights proportional to "}<InlineMath>{"[9,1,1]"}</InlineMath>{". Asking for A gives"}</Prose>

<div className="neural-equation"><MathBlock>{"\\alpha=[9/11,\\;1/11,\\;1/11],\\qquad\no=\\sum_j\\alpha_jv_j=\\frac{9(2)+7+4}{11}=\\frac{29}{11}\\approx2.636364."}</MathBlock></div>

<Prose>{"This is already an important correction to the phrase “exact retrieval.” We computed softmax attention exactly, but its answer is a mixture, not the original value 2. Increasing the score gap makes it concentrate more strongly on A. Equal or misleading keys, finite precision and learned projections can still prevent the intended retrieval."}</Prose>

<JambaWorkedReadFigure/>

<Prose>{"A hybrid can make both kinds of information available to later computation. It does not have to average these two answers together. For example, an output head could report a recent level and the value associated with a requested label as separate fields."}</Prose>

<H3>{"A specific collision explains the limitation"}</H3>

<Prose>{"Replace the values "}<InlineMath>{"[2,7,4]"}</InlineMath>{" by "}<InlineMath>{"[6,5,4]"}</InlineMath>{". Our recurrence again ends at "}<InlineMath>{"(s,n)=(8,1.75)"}</InlineMath>{". Once these two histories have collapsed to the same state, any later computation receiving only that state must give the same result for both histories. The label-A attention read instead changes from "}<InlineMath>{"29/11"}</InlineMath>{" to "}<InlineMath>{"63/11"}</InlineMath>{"."}</Prose>

<Prose>{"This is a counterexample for "}<strong>{"this two-number summary"}</strong>{", not a proof that every recurrent network fails at copying. A larger state, a selective write rule, a different representation or explicit training may preserve the required information. Conversely, retaining separate keys and values does not make a trained attention model infallible."}</Prose>

<JambaCollisionFigure/>

<HybridMemoryLab/>

<H2>{"2. Read a hybrid stack on two axes"}</H2>

<Prose>{"Attention and Mamba are "}<strong>{"sequence mixers"}</strong>{": an output at one position can depend on earlier positions. A feed-forward network applies a nonlinear transformation to each position's current vector. It can use context already incorporated into that vector, but it does not independently scan the other positions."}</Prose>

<Prose>{"A common layer has two residual updates:"}</Prose>

<div className="neural-equation"><MathBlock>{"u^\\ell=x^\\ell+\\operatorname{Mixer}_\\ell(\\operatorname{RMSNorm}(x^\\ell)),\n\\qquad\nx^{\\ell+1}=u^\\ell+\\operatorname{FFN}_\\ell(\\operatorname{RMSNorm}(u^\\ell))."}</MathBlock></div>

<Prose>{"Both additions matter. The second adds its result to "}<InlineMath>{"u^\\ell"}</InlineMath>{", which already contains the mixer's contribution. RMSNorm rescales each vector using its root-mean-square magnitude and learned channel weights; it does not subtract a mean or erase sequence order."}</Prose>

<JambaResidualFigure/>

<Prose>{"In the original Jamba configuration, zero-based layer indices 4, 12, 20 and 28 use attention. The other 28 use Mamba. Every odd-indexed layer uses 16 feed-forward experts, selecting two per token; the even-indexed layers use a single feed-forward network. Consequently, this released pattern's attention layers have dense FFNs. The architecture permits other combinations, but “could combine” and “does combine in this checkpoint” are different statements. These values come from the "}<a href={"https://huggingface.co/ai21labs/Jamba-v0.1/blob/main/config.json"}>{"released configuration"}</a>{"."}</Prose>

<NeuralTable caption={"2. Read a hybrid stack on two axes"} headers={[<>{"Layer indices in the first cycle"}</>,<>{"Mixer"}</>,<>{"Feed-forward operation"}</>]} rows={[[<>{"0, 2, 6"}</>,<>{"Mamba"}</>,<>{"Dense"}</>],[<>{"1, 3, 5, 7"}</>,<>{"Mamba"}</>,<>{"Top-2 of 16 experts"}</>],[<>{"4"}</>,<>{"Attention"}</>,<>{"Dense"}</>]]} />

<Prose>{"The cycle repeats across depth, not every eight input tokens. Every token passes through every layer. At an attention layer, its query can read earlier positions of "}<strong>{"that layer's input representation"}</strong>{"; those representations have already passed through lower layers."}</Prose>

<JambaDepthFigure/>

<Prose>{"Do earlier attention layers have “nothing to attend to”? No. Even the first layer receives representations of all available input tokens. Deeper layers may provide more contextualized representations, but useful placement is an empirical design choice. Nor is the final attention layer wasted: its outputs can directly influence the prediction head."}</Prose>

<Prose>{"The original paper's ratio comparison used 1.3B-parameter models trained on 250B tokens. Its 1:3 and 1:7 hybrids had similar results in that experiment, motivating the cheaper tested ratio. That is a scoped observation, not a theorem prescribing one attention layer per eight layers for every dataset, scale and implementation. "}<a href={"https://arxiv.org/html/2403.19887v1#S6.SS1"}>{"Jamba, §6.1"}</a>{""}</Prose>

<H2>{"3. What the Mamba and attention layers actually carry"}</H2>

<H3>{"Mamba: a selective update, followed by a read"}</H3>

<Prose>{"Recall the "}<a href={"/learn/path/full-curriculum/state-space-models-s4-mamba-mamba-2?module=deep-learning-fundamentals"}>{"state-space lesson"}</a>{": an input-dependent recurrence can choose how strongly to retain old state, write new information and read it back. In a Mamba-1-style channel, a simplified indexing of the implemented update is"}</Prose>

<div className="neural-equation"><MathBlock>{"H_{t,i,n}\n=e^{-\\Delta_{t,i}e^{a_{i,n}}}H_{t-1,i,n}\n+\\Delta_{t,i}B_{t,n}u_{t,i},\n\\qquad\ny_{t,i}=\\sum_nC_{t,n}H_{t,i,n}+D_i u_{t,i}."}</MathBlock></div>

<Prose>{"Here "}<InlineMath>{"i"}</InlineMath>{" indexes the expanded channels and "}<InlineMath>{"n"}</InlineMath>{" indexes state coordinates within a channel. "}<InlineMath>{"\\Delta"}</InlineMath>{" is positive; "}<InlineMath>{"e^{a_{i,n}}"}</InlineMath>{" is a learned positive decay rate. "}<InlineMath>{"B_t"}</InlineMath>{" and "}<InlineMath>{"C_t"}</InlineMath>{" depend on the current transformed input. This update is affine in the old state once the current input and coefficients are fixed, while the entire input-to-output model remains nonlinear."}</Prose>

<Prose>{"The input "}<InlineMath>{"u_t"}</InlineMath>{" has already passed through a learned projection, a short causal depthwise convolution and an activation. The read is gated and projected back into the residual stream. A width-4 convolution needs recent projected inputs too: retaining the recurrent matrix while discarding the convolution history does not preserve the layer's computation."}</Prose>

<JambaStateFigure/>

<Prose>{"Jamba adds RMSNorm to the low-rank timestep features and to "}<InlineMath>{"B"}</InlineMath>{" and "}<InlineMath>{"C"}</InlineMath>{" before the selective scan. The timestep features are subsequently projected and passed through softplus. Normalizing a final residual output is not an equivalent substitution. The "}<a href={"https://github.com/huggingface/transformers/blob/main/src/transformers/models/jamba/modeling_jamba.py"}>{"official implementation"}</a>{" also distinguishes full-sequence computation from state updates during decoding."}</Prose>

<Prose>{"This recurrence uses the common Mamba input discretization "}<InlineMath>{"\\Delta B u"}</InlineMath>{". It should not be silently substituted for the exact zero-order-hold expression of an arbitrary continuous-time system. The earlier state-space lesson derives that distinction."}</Prose>

<Prose>{"With fixed channel and state dimensions, each recurrent update has a fixed amount of work, so processing "}<InlineMath>{"T"}</InlineMath>{" inputs this way requires work proportional to "}<InlineMath>{"T"}</InlineMath>{". Projection and FFN work must still be included. The teaching program uses an explicit loop; optimized implementations can exploit the associative composition of the state-update maps to parallelize a scan. A Python loop's timing therefore does not predict an optimized Mamba kernel's timing."}</Prose>

<JambaSelectivePipeline/><JambaScanFigure/>

<H3>{"Attention: retain separate projected keys and values"}</H3>

<Prose>{"For a causal head, position "}<InlineMath>{"t"}</InlineMath>{" forms a query "}<InlineMath>{"q_t"}</InlineMath>{". Each permitted position "}<InlineMath>{"j\\le t"}</InlineMath>{" supplies a key "}<InlineMath>{"k_j"}</InlineMath>{" and a value "}<InlineMath>{"v_j"}</InlineMath>{":"}</Prose>

<div className="neural-equation"><MathBlock>{"\\alpha_{t,j}\n=\\frac{\\exp(q_t^\\top k_j/\\sqrt{d_h})}\n{\\sum_{r\\le t}\\exp(q_t^\\top k_r/\\sqrt{d_h})},\n\\qquad\no_t=\\sum_{j\\le t}\\alpha_{t,j}v_j."}</MathBlock></div>

<Prose>{"The causal mask excludes future positions. It does not exclude the current position in the convention used here. During generation, caching prior keys and values avoids computing their projections again. The new query still has to interact with the retained keys and values."}</Prose>

<Prose>{"Grouped-query attention lets several query heads share each key/value head. In the original Jamba configuration, 32 query heads share eight key/value heads, with width 128 per head. Cache storage depends on "}<strong>{"eight"}</strong>{", while the attention score calculations still serve "}<strong>{"32"}</strong>{" query heads. See "}<a href={"/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals"}>{"multi-head attention"}</a>{" for the full head construction."}</Prose>

<JambaHeadsFigure/>

<H3>{"Where does order enter?"}</H3>

<Prose>{"Jamba does not use RoPE or another explicit positional embedding in its original release. Its recurrent and causal-convolution operations are order-sensitive, so the later attention layers receive representations that can depend on preceding order. The original paper tested RoPE and no-RoPE variants; this does not establish that positional mechanisms are unnecessary in every hybrid."}</Prose>

<Prose>{"For example, processing A then B through a recurrence generally differs from processing B then A. An attention layer can operate on those different contextualized vectors even without a separately added position vector. This is different from declaring that an unordered collection of raw token embeddings contains order information."}</Prose>

<Prose>{"The model below deliberately includes simple position tags in "}<strong>{"all"}</strong>{" its neural candidates, including the all-attention baseline. That makes its local comparison easier to interpret; it is one reason it is a teaching model rather than a Jamba checkpoint replica."}</Prose>

<H2>{"4. Calculate the cache before making a hardware claim"}</H2>

<Prose>{"Separate model weights, persistent request state and temporary working memory. A small KV cache says nothing by itself about whether hundreds of billions of weights fit on a device."}</Prose>

<Prose>{"For batch size "}<InlineMath>{"B"}</InlineMath>{", context length "}<InlineMath>{"T"}</InlineMath>{", "}<InlineMath>{"L_A"}</InlineMath>{" full-attention layers, "}<InlineMath>{"H_{KV}"}</InlineMath>{" key/value heads, head width "}<InlineMath>{"d_h"}</InlineMath>{" and "}<InlineMath>{"b_{KV}"}</InlineMath>{" bytes per stored scalar,"}</Prose>

<div className="neural-equation"><MathBlock>{"M_{KV}=B L_A T\\,2H_{KV}d_h b_{KV}."}</MathBlock></div>

<Prose>{"The factor two represents keys and values. For "}<InlineMath>{"L_M"}</InlineMath>{" Mamba layers with expanded width "}<InlineMath>{"I"}</InlineMath>{", state width "}<InlineMath>{"N"}</InlineMath>{" and convolution-buffer width "}<InlineMath>{"C"}</InlineMath>{","}</Prose>

<div className="neural-equation"><MathBlock>{"M_{\\mathrm{rec}}=B L_M I N b_{\\mathrm{rec}},\n\\qquad\nM_{\\mathrm{conv}}=B L_M I C b_{\\mathrm{conv}}."}</MathBlock></div>

<Prose>{"The exact buffer layout is an implementation choice. Our accounting reserves "}<InlineMath>{"C"}</InlineMath>{" projected inputs per channel, a common update-buffer layout; a minimal mathematical representation could retain only the previous "}<InlineMath>{"C-1"}</InlineMath>{". State and convolution dtypes must be stated separately."}</Prose>

<Prose>{"Consider the original 32-layer shape: "}<InlineMath>{"L_A=4"}</InlineMath>{", "}<InlineMath>{"L_M=28"}</InlineMath>{", "}<InlineMath>{"I=8192"}</InlineMath>{", "}<InlineMath>{"N=16"}</InlineMath>{", "}<InlineMath>{"C=4"}</InlineMath>{", eight KV heads and head width 128. Assume batch one, 2-byte K/V and convolution storage, and 4-byte recurrent state. Each attention layer stores 4,096 bytes per token. Across Mamba layers, recurrent state takes 14 MiB and convolution buffers take 1.75 MiB."}</Prose>

<NeuralTable caption={"4. Calculate the cache before making a hardware claim"} headers={[<>{"Context tokens"}</>,<>{"All-attention 32-layer cache"}</>,<>{"Hybrid K/V"}</>,<>{"Hybrid recurrent + convolution"}</>,<>{"Hybrid total"}</>]} rows={[[<>{"1,024"}</>,<>{"0.125 GiB"}</>,<>{"0.015625 GiB"}</>,<>{"0.015381 GiB"}</>,<>{"0.031006 GiB"}</>],[<>{"4,096"}</>,<>{"0.5 GiB"}</>,<>{"0.0625 GiB"}</>,<>{"0.015381 GiB"}</>,<>{"0.077881 GiB"}</>],[<>{"16,384"}</>,<>{"2 GiB"}</>,<>{"0.25 GiB"}</>,<>{"0.015381 GiB"}</>,<>{"0.265381 GiB"}</>],[<>{"65,536"}</>,<>{"8 GiB"}</>,<>{"1 GiB"}</>,<>{"0.015381 GiB"}</>,<>{"1.015381 GiB"}</>],[<>{"262,144"}</>,<>{"32 GiB"}</>,<>{"4 GiB"}</>,<>{"0.015381 GiB"}</>,<>{"4.015381 GiB"}</>]]} />

<Prose>{"These are calculated tensor sizes, not measured GPU allocations. GiB means "}<InlineMath>{"2^{30}"}</InlineMath>{" bytes; GB means "}<InlineMath>{"10^9"}</InlineMath>{" bytes. The comparison keeps head configuration and depth fixed. It makes no assertion that the two networks have equal parameter count or predictive quality."}</Prose>

<JambaMemoryFigure/>

<Prose>{"The K/V component is exactly eight times smaller in this comparison. The total cache is not exactly eight times smaller because the hybrid also carries recurrent state. At zero processed tokens, our preallocated hybrid buffers already have a nonzero size; the all-attention K/V count is zero. Allocators may reserve memory differently."}</Prose>

<Prose>{"Full attention remains present, so this hybrid's cache still grows with context length. Replacing all its attention layers with a fixed-size sliding window would eventually bound the K/V count too, but would remove direct reads of keys outside that window. That is a changed computation, not a free cache optimization."}</Prose>

<HybridBudgetLab/>

<H3>{"Optional: prefill arithmetic is different from one-token decoding"}</H3>

<Prose>{"Prefill processes the supplied prompt. Decode produces a new token using the accumulated state. Ignoring bias additions, normalization, softmax and implementation padding, and counting a multiply and add as two operations, attention projections cost"}</Prose>

<div className="neural-equation"><MathBlock>{"4BTd(d+H_{KV}d_h)"}</MathBlock></div>

<Prose>{"when query and output widths both equal "}<InlineMath>{"d"}</InlineMath>{". The two attention matrix products over an ideally evaluated causal triangle cost"}</Prose>

<div className="neural-equation"><MathBlock>{"4Bd\\frac{T(T+1)}{2}=2BdT(T+1)."}</MathBlock></div>

<Prose>{"A single new token after "}<InlineMath>{"T"}</InlineMath>{" retained positions instead requires approximately "}<InlineMath>{"4Bd(T+1)"}</InlineMath>{" pair-product operations, plus its projections. The history-dependent term is linear per new token and quadratic when accumulating a whole full-attention sequence. Actual kernels may evaluate additional tiles."}</Prose>

<Prose>{"At "}<InlineMath>{"T=32,768"}</InlineMath>{", "}<InlineMath>{"d=4,096"}</InlineMath>{", "}<InlineMath>{"H_{KV}d_h=1,024"}</InlineMath>{" and "}<InlineMath>{"B=1"}</InlineMath>{", one attention layer's projections contribute "}<InlineMath>{"2.748779\\times10^{12}"}</InlineMath>{" operations and the ideal causal pair products contribute "}<InlineMath>{"8.796361\\times10^{12}"}</InlineMath>{". These are "}<strong>{"per prompt"}</strong>{", not per token. The recurrence and FFN costs also matter; the table is not a complete model FLOP estimate or a latency forecast."}</Prose>

<JambaArithmeticFigure/>

<H2>{"5. Experts are a separate architectural choice"}</H2>

<Prose>{"At an MoE feed-forward layer, a router scores experts using the current token vector. Only selected experts transform that vector. This changes which "}<strong>{"parameters"}</strong>{" participate; it does not select which earlier "}<strong>{"tokens"}</strong>{" attention reads."}</Prose>

<Prose>{"For a SwiGLU expert,"}</Prose>

<div className="neural-equation"><MathBlock>{"F_e(x)=W_{\\mathrm{down},e}\n\\big[\\operatorname{SiLU}(W_{\\mathrm{gate},e}x)\n\\odot W_{\\mathrm{up},e}x\\big]."}</MathBlock></div>

<Prose>{"Each of its three matrices contributes "}<InlineMath>{"df"}</InlineMath>{" parameters when the model width is "}<InlineMath>{"d"}</InlineMath>{" and intermediate width is "}<InlineMath>{"f"}</InlineMath>{". Ignoring biases, that is "}<InlineMath>{"3df"}</InlineMath>{" parameters per expert. The gating branch is why a two-matrix estimate is wrong here."}</Prose>

<Prose>{"Let "}<InlineMath>{"p=\\operatorname{softmax}(r(x))"}</InlineMath>{" and let "}<InlineMath>{"S"}</InlineMath>{" contain the top two indices. In the inspected Jamba implementation, the contribution is"}</Prose>

<div className="neural-equation"><MathBlock>{"z(x)=\\sum_{e\\in S}p_e F_e(x)."}</MathBlock></div>

<Prose>{"The selected weights retain their probabilities from the full softmax; they are not divided by their selected sum. Other MoE implementations use that additional renormalization. The distinction changes the function and must match the checkpoint."}</Prose>

<JambaRouterWorked/>

<Prose>{"For scores "}<InlineMath>{"[\\log4,\\log2,0,0]"}</InlineMath>{", probabilities are "}<InlineMath>{"[1/2,1/4,1/8,1/8]"}</InlineMath>{". If the first two expert outputs are "}<InlineMath>{"[2,-1]"}</InlineMath>{" and "}<InlineMath>{"[0,3]"}</InlineMath>{", Jamba-style mixing gives"}</Prose>

<div className="neural-equation"><MathBlock>{"\\tfrac12[2,-1]+\\tfrac14[0,3]=[1,\\tfrac14]."}</MathBlock></div>

<Prose>{"Renormalizing the selected weights would give "}<InlineMath>{"[4/3,1/3]"}</InlineMath>{". Neither convention can be inferred merely from the phrase “top-2.”"}</Prose>

<HybridRouterLab/>

<Prose>{"The broader "}<a href={"/learn/path/full-curriculum/mixture-of-experts-transformers-moe?module=deep-learning-fundamentals"}>{"Mixture-of-Experts lesson"}</a>{" develops expert specialization, balancing losses, capacity, dispatch and training gradients. Locally, remember that storing 16 experts and executing two does not make 14 experts disappear from model memory. Across a batch, different tokens may activate many different experts, requiring weight movement and possibly communication among devices."}</Prose>

<H3>{"Optional: count a real-shaped FFN pool"}</H3>

<Prose>{"Using "}<InlineMath>{"d=4,096"}</InlineMath>{", "}<InlineMath>{"f=14,336"}</InlineMath>{", 16 dense FFNs and 16 MoE FFNs with 16 experts each:"}</Prose>

<NeuralTable caption={"Optional: count a real-shaped FFN pool"} headers={[<>{"Quantity"}</>,<>{"Calculation"}</>,<>{"Parameters"}</>]} rows={[[<>{"One SwiGLU expert"}</>,<>{""}<InlineMath>{"3df"}</InlineMath>{""}</>,<>{"176,160,768"}</>],[<>{"All FFN weights"}</>,<>{""}<InlineMath>{"(16+16\\cdot16)3df"}</InlineMath>{""}</>,<>{"47,915,728,896"}</>],[<>{"FFN weights active for one token"}</>,<>{""}<InlineMath>{"(16+16\\cdot2)3df"}</InlineMath>{""}</>,<>{"8,455,716,864"}</>],[<>{"Hypothetical 32 dense FFNs"}</>,<>{""}<InlineMath>{"32\\cdot3df"}</InlineMath>{""}</>,<>{"5,637,144,576"}</>]]} />

<Prose>{"The FFN pool is 8.5 times the all-dense pool, while active FFN parameters are 1.5 times as many. These counts exclude mixers, embeddings, routers and normalization. They explain how the original model can have roughly 52B total and 12B active parameters, without inventing a claim that sparse capacity is free."}</Prose>

<JambaParameterFigure/>

<H2>{"6. Distinguish the family from a particular release"}</H2>

<NeuralTable caption={"6. Distinguish the family from a particular release"} headers={[<>{"Release"}</>,<>{"Total / active parameters, reported approximately"}</>,<>{"Depth and width"}</>,<>{"Architecture note"}</>]} rows={[[<>{"Jamba-v0.1"}</>,<>{"52B / 12B"}</>,<>{"32 layers, width 4,096"}</>,<>{"Original base model; four attention layers"}</>],[<>{"Jamba-1.5 Mini"}</>,<>{"52B / 12B"}</>,<>{"Same model-size family"}</>,<>{"Updated, instruction-tuned successor"}</>],[<>{"Jamba-1.5 Large"}</>,<>{"398B / 94B"}</>,<>{"72 layers, width 8,192"}</>,<>{"Nine eight-layer cycles, nine attention layers"}</>]]} />

<Prose>{"The larger model is not the 32-layer configuration with a larger label attached. Its 64 query heads and eight KV heads also differ from the original model's 32 query heads. Jamba-1.5 retained Mamba-1 after its authors compared Mamba-1 and Mamba-2 hybrids; a newer mixer was not automatically better in their tested combination. "}<a href={"https://arxiv.org/html/2408.12570v1#S2"}>{"Jamba-1.5, §2"}</a>{""}</Prose>

<Prose>{"At 52B parameters, two bytes per weight alone is approximately "}<strong>{"104 billion bytes"}</strong>{", before caches and workspaces. An original single-80GB deployment discussion depended on quantization. The 1.5 paper's ExpertsInt8 technique stores selected feed-forward weights in int8 and dequantizes within the compute kernel. It does not turn the entire model into a uniformly int8 network or mean that only active expert weights require storage."}</Prose>

<JambaPrecisionFigure/>

<Prose>{"There is also a numerical-range issue, separate from byte counts. The largest finite float16 value is 65,504; bfloat16 has a much wider exponent range. A computation that remains finite in bfloat16 can therefore overflow after an inference system switches its activations to float16. The Jamba-1.5 authors reported large internal activations and added a small penalty proportional to their mean squared magnitude. This is an example of adapting training to a serving constraint, not a universal requirement to add the same coefficient to every hybrid. "}<a href={"https://arxiv.org/html/2408.12570v1#S3.SS2"}>{"Jamba-1.5, §3.2"}</a>{""}</Prose>

<Prose>{"For later releases, inspect the exact model card and configuration again. The "}<a href={"https://huggingface.co/ai21labs/AI21-Jamba-Mini-1.7"}>{"Jamba Mini 1.7 card"}</a>{", checked on 13 September 2026, describes a 256K context and instruction/grounding updates, and explicitly distinguishes its multi-GPU BF16 requirements. Its example model identifiers and repository title are not consistently ordered, so verify the resolvable identifier before copying a deployment command."}</Prose>

<JambaReleaseFigure/>

<Prose>{"Model capabilities require training too. Jamba-1.5 describes pretraining, a stage emphasizing long documents, and post-training that mixes skill data with long-context examples. A maximum input-length field, or the ability to allocate a cache that large, does not prove reliable reasoning throughout that context."}</Prose>

<H2>{"7. Train and inspect a small hybrid on real pen trajectories"}</H2>

<Prose>{"The "}<a href={"https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits"}>{"UCI PenDigits dataset"}</a>{" contains digit traces collected from 44 writers with a tablet. Each released example contains eight "}<InlineMath>{"(x,y)"}</InlineMath>{" points and a digit label. These points were "}<strong>{"resampled at approximately regular distance along the stroke"}</strong>{", not at regular time intervals. Connecting them produces a path; treating them as an image grid or attaching a time-in-seconds axis would misrepresent the data."}</Prose>

<Prose>{"The original dataset provides 7,494 development rows from 30 writers and 3,498 assessment rows from 14 different writers. Individual writer IDs are not supplied in these flat files. We preserve the official boundary, then use a deterministic classwise shuffle within the development file to select 100 fitting and 30 validation rows per digit. The remaining development rows stay unused. All 10,992 coordinate sequences are distinct in an exact-feature check, with no conflicting labels or cross-file duplicates."}</Prose>

<JambaWorkedStrokeFigure/>

<Prose>{"We divide coordinates by 50 and subtract one, giving values in "}<InlineMath>{"[-1,1]"}</InlineMath>{". This fixed transformation learns nothing from validation or assessment rows. All neural candidates receive the same four features at position "}<InlineMath>{"t=0,\\ldots,7"}</InlineMath>{":"}</Prose>

<div className="neural-equation"><MathBlock>{"[x_t/50-1,\\;y_t/50-1,\\;t/7,\\;(t/7)^2]."}</MathBlock></div>

<Prose>{"The position tags are fixed to the original eight-point grid. They must not be rescaled when inspecting only a prefix or restarted at a chunk boundary."}</Prose>

<H3>{"The model and protocol"}</H3>

<Prose>{"The neural models embed each four-feature vector into width 16 and pass it through three causal residual layers. M denotes a selective recurrent mixer with state width four and a width-3 causal convolution. A denotes one causal attention head. Every layer has a dense SwiGLU FFN with intermediate width 32. A final RMSNorm and linear classifier read the last position and produce ten logits."}</Prose>

<Prose>{"The recurrent mixer includes input-dependent writes, reads and positive timesteps, plus a learned decay, a skip term and an output gate. It omits Jamba's expansion and low-rank timestep factorization. The attention has one head rather than GQA. These simplifications keep every state update inspectable. No MoE is trained in this experiment; expert routing was isolated in §5."}</Prose>

<JambaPositionFigure/>

<JambaCandidatesFigure/>

<Prose>{"The fitting objective is cross-entropy:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\mathcal L=-\\frac1m\\sum_{i=1}^{m}\n\\log\\operatorname{softmax}(z_i)_{y_i}."}</MathBlock></div>

<Prose>{"All runs use 80 epochs, batches of 100, Adam at learning rate 0.003, gradient-norm clipping at one and the same sequence of batch permutations. Each run selects its lowest validation cross-entropy checkpoint, retaining the earliest checkpoint if losses tie. We predeclare four neural patterns at seed 37, repeat MAM at seed 73, and include a 170-parameter linear classifier on the flattened 16 coordinates. A matching depth does not imply matching parameter counts."}</Prose>

<Prose>{"The complete "}<a href={"/learn-assets/hybrid-ssm-transformer-architectures-jamba/stroke_models.py"}>{"training and model program"}</a>{", "}<a href={"/learn-assets/hybrid-ssm-transformer-architectures-jamba/data-provenance.md"}>{"input data and provenance"}</a>{", saved fitted arrays and result files accompany this lesson. In a separate Python environment with NumPy and PyTorch installed, keep the companion files in one directory and run:"}</Prose>

<CodeBlock language={"sh"}>{"python stroke_models.py\npython inspect_stroke.py --row 2452 --break-at 3 --mode carry\npython inspect_stroke.py --row 2452 --break-at 3 --mode convolution-reset"}</CodeBlock>

<Prose>{"The first command prints saved results without training. To reproduce all six fits, run "}<code>{"python stroke_models.py --train"}</code>{". This author run used CPU PyTorch 2.14.0 and NumPy 2.3.5. No foundation-model weights or GPU kernels are required for these companions."}</Prose>

<Prose>{"Here is a complete small inspection program using the supplied modules:"}</Prose>

<CodeBlock language={"python"}>{"import numpy as np\nfrom stroke_models import HERE\nfrom inspect_stroke import run_trace\n\nrows = np.loadtxt(HERE / \"pendigits.tra\", delimiter=\",\")\ncoordinates = rows[2452 - 1, :16].reshape(8, 2)\n\nfor mode in (\"carry\", \"recurrent-reset\", \"kv-reset\", \"convolution-reset\"):\n    result = run_trace(coordinates, key=\"MAM-37\", break_at=3, mode=mode)\n    print(\n        mode,\n        result[\"branch_prediction\"],\n        round(result[\"branch_probabilities\"][0], 6),\n        round(result[\"maximum_logit_difference\"], 6),\n    )"}</CodeBlock>

<Prose>{"It reports the predicted digit, probability assigned to zero and largest difference from uninterrupted logits. “Recurrent reset” clears both the recurrent state and that mixer's convolution history; “convolution reset” isolates the latter."}</Prose>

<Prose>{"The key residual update in the full program is short enough to inspect directly:"}</Prose>

<CodeBlock language={"python"}>{"def forward(self, x):\n    x = x + self.mixer(self.norm1(x))\n    return x + self.channel(x)"}</CodeBlock>

<Prose>{"The channel method normalizes its input, projects separate gate/up branches, multiplies the SiLU gate by the up branch and projects the result back. The selective mixer's step method returns both its output and its updated pair of caches. The attention step appends K/V tensors and reads the complete permitted bank. Thus training, full-sequence inference and streaming implement the same intended function."}</Prose>

<JambaProgram file="stroke_models.py"/><JambaProgram file="inspect_stroke.py"/>

<H3>{"Results, including the uncomfortable comparisons"}</H3>

<NeuralTable caption={"Results, including the uncomfortable comparisons"} headers={[<>{"Candidate"}</>,<>{"Parameters"}</>,<>{"Selected epoch"}</>,<>{"Fit errors / 1,000"}</>,<>{"Validation errors / 300"}</>,<>{"Assessment errors / 3,498"}</>]} rows={[[<>{"Flattened linear, seed 37"}</>,<>{"170"}</>,<>{"80"}</>,<>{"114"}</>,<>{"30"}</>,<>{"565"}</>],[<>{"MMM, seed 37"}</>,<>{"9,314"}</>,<>{"79"}</>,<>{"2"}</>,<>{"7"}</>,<>{"201"}</>],[<>{"AAA, seed 37"}</>,<>{"8,474"}</>,<>{"64"}</>,<>{"3"}</>,<>{"12"}</>,<>{"167"}</>],[<>{"MAM, seed 37"}</>,<>{"9,034"}</>,<>{"56"}</>,<>{"0"}</>,<>{"8"}</>,<>{"193"}</>],[<>{"AMM, seed 37"}</>,<>{"9,034"}</>,<>{"68"}</>,<>{"6"}</>,<>{"12"}</>,<>{"261"}</>],[<>{"MAM, seed 73"}</>,<>{"9,034"}</>,<>{"78"}</>,<>{"2"}</>,<>{"9"}</>,<>{"183"}</>]]} />

<Prose>{"These are actual runs. MMM has the fewest validation classification errors, while AAA has the fewest assessment errors in this table. MAM is not the overall winner. Its second seed also differs from its first. We do not select an architecture retrospectively using the assessment column."}</Prose>

<Prose>{"The comparison supports several useful conclusions. Learned sequence models can recognize these real trajectories. The arrangement of mixers can change fitted behavior even at equal parameter count: MAM and AMM differ. A hybrid need not beat every simpler model, and performance on eight-point handwriting does not establish language-model retrieval or 256K-context throughput."}</Prose>

<Prose>{"Selection used validation "}<strong>{"cross-entropy"}</strong>{", not the displayed error count. That distinction matters when a model becomes more confident on already-correct examples or makes a few highly confident mistakes. Keep both measures when interpreting learning curves."}</Prose>

<JambaTrainingFigure/>

<H3>{"Follow a prediction through a real cache"}</H3>

<Prose>{"At a three-point boundary, the MAM model carries:"}</Prose>

<NeuralTable caption={"Follow a prediction through a real cache"} headers={[<>{"Layer"}</>,<>{"Persistent object for one request"}</>,<>{"Elements"}</>]} rows={[[<>{"M0"}</>,<>{"State "}<InlineMath>{"16\\times4"}</InlineMath>{"; prior projected inputs "}<InlineMath>{"16\\times2"}</InlineMath>{""}</>,<>{"64 + 32"}</>],[<>{"A1"}</>,<>{"Three keys and three values, each width 16"}</>,<>{"96"}</>],[<>{"M2"}</>,<>{"State "}<InlineMath>{"16\\times4"}</InlineMath>{"; prior projected inputs "}<InlineMath>{"16\\times2"}</InlineMath>{""}</>,<>{"64 + 32"}</>]]} />

<Prose>{"Our tiny convolution implementation retains the previous two inputs, then appends the new one; its cache layout therefore differs from the four-slot accounting convention in §4. Count the representation that the program actually stores."}</Prose>

<Prose>{"For validation source row 2,452, labelled zero, the uninterrupted model assigns probability 0.997078 to zero. Carrying all caches reproduces it to floating-point tolerance. Clearing convolution history at the boundary changes intermediate logits by about 9.03, yet the final predicted digit remains zero. Checking only the final label would miss that serious computation change."}</Prose>

<Prose>{"Clearing K/V also changes this example's logits while preserving its final class. A changed class is strong evidence of a behavioral difference; an unchanged class is not evidence of equivalence."}</Prose>

<JambaContinuationFigure/>

<HybridStrokeLab/>

<Prose>{"The saved model was trained to classify after all eight points. Earlier logits reveal its computation, but are not calibrated promises that it can reliably classify every partial stroke."}</Prose>

<H2>{"8. Preserve request identity when serving"}</H2>

<Prose>{"A recurrent state is specific to the prefix, model weights, layer and request that produced it. It is not a global scratch buffer that can be handed to the next user."}</Prose>

<Prose>{"Suppose requests R and S arrive interleaved. To process the next token of R, the server needs R's state and convolution history at every recurrent layer, R's K/V banks at every attention layer, and the correct processed-token count. Reusing S's state can produce a plausible-looking answer that depends on the wrong input."}</Prose>

<JambaRequestsFigure/>

<Prose>{"Prefill may be divided into chunks. If a prefix has already been processed, the next chunk must continue from all required state objects. The small model verifies this by comparing full-sequence and token/chunk paths, including gradients. Its largest full-versus-stream output discrepancy across the five neural models' 300 validation sequences was about "}<InlineMath>{"1.26\\times10^{-5}"}</InlineMath>{" in float32. A separate two-sequence float64 MAM check gave output discrepancy "}<InlineMath>{"1.42\\times10^{-14}"}</InlineMath>{" and parameter-gradient discrepancy below "}<InlineMath>{"3.1\\times10^{-15}"}</InlineMath>{". These are author calculations for this small implementation."}</Prose>

<Prose>{"Prefix reuse needs more than matching text. The cached state must correspond to the same model revision and any adapters, the same relevant tokenization, the same valid prefix boundary and all necessary layer types. To branch generation from a prefix, preserve an immutable snapshot or use copy-on-write; do not let the first continuation overwrite the second continuation's starting state."}</Prose>

<Prose>{"Rewinding is different for the two storage forms. Removing the final K/V entries can undo their append operation. A recurrent update is not generally invertible: decay, selective writes and finite precision may have lost information. To rewind, restore an earlier snapshot or recompute the prefix. Speculative decoding, beam branching and editing previous input all need this distinction."}</Prose>

<JambaRewindFigure/>

<Prose>{"The "}<a href={"https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/"}>{"vLLM hybrid-cache design document"}</a>{" explains why allocation and prefix-hit rules differ by layer type. Its described algorithm is tied to a named implementation revision and labels parts of Mamba prefix caching as work in progress. Treat it as an explanation of the engineering problem; check the installed release's actual support before depending on a particular optimization."}</Prose>

<H3>{"A deployment experiment that answers a real question"}</H3>

<Prose>{"Begin with a specific workload: perhaps searching a collection of incident reports for all references to a failing component and returning cited evidence. Decide what counts as a correct answer before comparing models."}</Prose>

<Prose>{"Use prompts that vary independently in length, evidence position, number of relevant records and distractor similarity. Include missing-evidence questions, conflicting statements and queries requiring several records to be joined. A single inserted sentence that repeats the answer verbatim tests a useful ability, but not all of document reasoning."}</Prose>

<Prose>{"Measure time to first token, time per output token, total request latency, peak memory, and supported concurrent requests at fixed output length and sampling settings. Record the GPU count/model, dtypes, quantization, software revision, kernel path and batching policy. Separate a cold model load from warm serving."}</Prose>

<Prose>{"Compare models at a constraint you actually care about: acceptable task quality under a fixed memory budget, or throughput while meeting a latency limit. Equal active parameters, equal total parameters, equal training tokens and equal serving cost are different comparisons. No single invented “relative compute” curve can substitute for them."}</Prose>

<Prose>{"The optional "}<a href={"/learn-assets/hybrid-ssm-transformer-architectures-jamba/deployment_example.py"}>{"deployment program"}</a>{" shows a complete single-prompt generation path with an explicit model ID and revision, correct prompt handling and input/output token accounting. It is provided for a suitably provisioned environment; it was not executed for this lesson. The bounded author experiment uses only the small stroke models."}</Prose>

<JambaProgram file="deployment_example.py"/>

<Prose>{"For fine-tuning, inspect the real module names before selecting adapter targets. Attention projections, recurrent input/timestep/output projections and FFNs provide different adaptation choices. Attention-only adaptation is a valid restricted experiment, not automatically a bug; broader targets may help at additional cost. Compare on held-out tasks, including the original context-length requirements. There is no universal rule that recurrent layers require exactly twice the learning rate or a different clipping threshold."}</Prose>

<H3>{"Reuse primitives, own the composition and the cache"}</H3>

<Prose>{"The core implementation is "}<a href={"/learn-assets/hybrid-ssm-transformer-architectures-jamba/stroke_models.py"}>{"stroke_models.py"}</a>{". "}<code>{"SelectiveMixer.step"}</code>{" explicitly computes the positive step size, stable negative decay rates, selective writes/reads and causal convolution history. "}<code>{"AttentionMixer.forward"}</code>{" constructs masked scores; "}<code>{"AttentionMixer.step"}</code>{" extends the exact K/V cache. "}<code>{"Layer"}</code>{" composes the chosen mixer with the residual and gated channel path, and "}<code>{"StrokeModel"}</code>{" exposes both full and streamed execution. Thus the learner can build the hybrid's defining behavior without importing a ready-made Jamba block."}</Prose>

<Prose>{"These classes use ordinary "}<code>{"nn.Linear"}</code>{", "}<code>{"nn.Conv1d"}</code>{", tensor operations, parameter registration and Adam. The full study supplies fitting and "}<code>{"state_dict"}</code>{" restoration. For a released model rather than our small instructional network, "}<a href={"/learn-assets/hybrid-ssm-transformer-architectures-jamba/deployment_example.py"}>{"deployment_example.py"}</a>{" supplies the distinct Transformers tokenizer/model/generation route, with an explicit checkpoint revision and resource assumptions. Its learned projections, dimensions, routing and cache classes belong to that selected release; our tiny output is not a numerical oracle for an unrelated pretrained model. The deployment program remains unexecuted and requires appropriate CUDA, kernels and model storage."}</Prose>

<Prose>{"Underlying attention, SSM and expert derivations are taught in their named earlier lessons. This packet stays self-contained for its own simplified mixers and routing calculation. It does not claim to have recreated a fused Mamba kernel or trained a full Jamba MoE checkpoint. "}<code>{"hybrid_mechanisms.py::route"}</code>{" explains the selected-expert probability contract; sparse trainable expert dispatch is owned by "}<a href={"/learn-code/mixture-of-experts-transformers-moe/moe_study.py"}>{"the complete MoE program"}</a>{", not the dense channel network used in this stroke experiment."}</Prose>

<Prose>{""}<strong>{"Control request identity."}</strong>{" Run two distinct trajectory prefixes, save each cache and position offset, then continue each with its own suffix. Compare with independent unsplit passes. Swap only the caches, then restore the correct pair."}</Prose>

<details><summary>Hint and reasoned solution</summary>

<Prose>{"A request state comprises every layer's recurrent/convolution or K/V state plus its next logical position. Keep an independent cache container for each request; "}<code>{"stream"}</code>{" updates the supplied list, so a shallow alias shared by two requests is unsafe. Full and correctly carried execution should agree within the declared float32 tolerance. Swapping caches changes the past information, and restarting the position offset changes the embedding even if the past tensors were right. Winning labels can remain unchanged, so compare prefix logits and individual state arrays. Recurrent carried storage stays fixed at fixed dimensions; attention K/V storage grows with retained sequence length. The current repeated "}<code>{"torch.cat"}</code>{" in the small reference may copy the old cache; a production serving cache would preallocate or page it while preserving this identity contract."}</Prose>

</details>

<H2>{"9. Optional branches: other ways to combine memory"}</H2>

<H3>{"Share weights across depth: Zamba and Zamba2"}</H3>

<Prose>{"Zamba repeatedly invokes a shared attention/MLP module along a Mamba backbone. Its shared module receives a concatenation of the current residual representation and the original input embedding. “Shared” refers to parameters reused at several depths, not to applying attention only to every sixth token. "}<a href={"https://arxiv.org/html/2405.16712v1#S2"}>{"Zamba, §II"}</a>{""}</Prose>

<Prose>{"This can save parameter storage while still doing attention computation at each invocation. If two invocations receive different hidden vectors, the same projection weights can produce different keys and values. Weight sharing "}<strong>{"alone"}</strong>{" does not prove that their KV caches can be identified. An architecture must explicitly define any cache sharing."}</Prose>

<Prose>{"The "}<a href={"https://huggingface.co/Zyphra/Zamba2-7B"}>{"Zamba2 model card"}</a>{" describes Mamba-2, two alternating shared blocks, depth-specific low-rank adapters and RoPE in the shared attention. These are concrete design changes, not interchangeable names for the original Jamba pattern."}</Prose>

<H3>{"Use local attention: Griffin and Samba"}</H3>

<Prose>{""}<a href={"/learn/path/full-curriculum/long-context-sequence-models-transformer-xl-griffin-perceiver?module=deep-learning-fundamentals"}>{"Griffin"}</a>{" combines a gated linear recurrence with local attention. Samba combines Mamba, sliding-window attention and separate FFNs. Its attention directly reads the window; information from further back must arrive through recurrent state or representations propagated through the stack. A fixed window can bound attention storage during streaming."}</Prose>

<Prose>{"“The program can continue processing” and “the model can recover any detail from arbitrarily far back” are different claims. Samba's reported length-generalization and passkey experiments have particular data and training conditions. They do not establish unlimited accurate memory for arbitrary histories. "}<a href={"https://arxiv.org/html/2406.07522v1"}>{"Samba, §§2–3"}</a>{""}</Prose>

<H3>{"Fuse heads inside a layer: Hymba"}</H3>

<Prose>{"Hymba sends the same layer input to attention and SSM branches in parallel, normalizes and rescales their outputs, then combines them. It also uses combinations of local/global attention, explicit cross-layer KV sharing and learned prefix meta tokens. The latter can serve as learned cache initialization; they are not new external facts retrieved at inference time. "}<a href={"https://arxiv.org/html/2411.13676v1#S2"}>{"Hymba, §2"}</a>{""}</Prose>

<JambaSharedVariantsFigure/>

<Prose>{"A useful distinction is whether the second operation receives the first operation's transformed output, as in a serial composition, or both receive the same representation before fusion. Neither composition universally dominates the other; the information flow and parameter budget differ."}</Prose>

<JambaAlternativesFigure/>

<Prose>{"As a more recent deployment example, "}<a href={"https://www.ibm.com/granite/docs/models/granite4-0"}>{"IBM's Granite 4.0 documentation"}</a>{" distinguishes hybrid Mamba-2 variants, dense versus MoE variants, and traditional alternatives. The model suffix and configuration matter: an organization or family name does not mean every model uses an SSM."}</Prose>

<H2>{"10. Connect the mechanism to a useful application"}</H2>

<Prose>{"The handwriting example suggests a less obvious application of hybrid thinking: a stream can contain both an evolving shape and specific landmarks. A turn near the beginning may distinguish two otherwise similar strokes. A recurrent summary and an address-sensitive operation provide different routes for learning that distinction. Our measured results also show why the architecture still needs to earn its place against simpler baselines."}</Prose>

<Prose>{"For incident reports, a running representation could combine a chain of symptoms, while attention could make an earlier part number available when answering a question. To test that hypothesis, change only the part number while holding the story fixed, then change the causal ordering while holding the identifiers fixed. Measure whether the model responds to the information each question requires."}</Prose>

<Prose>{"For scientific notebooks, the needed link may be between a final conclusion and a parameter setting far earlier in the record. A useful evaluation asks for the setting "}<strong>{"and its evidence location"}</strong>{", includes repeated parameter names in unrelated experiments, and distinguishes the final setting from a superseded value. Increasing context capacity helps only if the model and application preserve those distinctions."}</Prose>

<Prose>{"These are application designs to investigate, not observed capabilities of our small classifier. For exact identifiers or auditable numerical records, an external database or structured retrieval step can be a better source of truth than asking any neural architecture to remember and regenerate the value unaided."}</Prose>

<Prose>{"The architecture decision can now be made as a sequence of practical questions:"}</Prose>

<ol start={1}><li>{"What must be preserved: a state summary, addressable details, local structure, or a combination?"}</li><li>{"Which operation provides a route for that information, and where in depth does it enter?"}</li><li>{"What grows with input length, batch size and model width?"}</li><li>{"Can the implementation carry every state correctly through the actual serving workflow?"}</li><li>{"Does the measured task improvement justify the memory, training and software cost?"}</li></ol>

<section className="lesson-ending lesson-ending--practice" data-lesson-ending="practice"><H2>{"11. Practice"}</H2>

<Prose>{"Attempt each question before opening its hint. Questions 1–6 cover the core route; 7–10 explore deeper architecture and engineering decisions."}</Prose>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"1. An identical summary, a different answer"}</H3>

<Prose>{"With "}<InlineMath>{"\\lambda=1/2"}</InlineMath>{", do "}<InlineMath>{"[1,4,2]"}</InlineMath>{" and "}<InlineMath>{"[5,2,2]"}</InlineMath>{" end at the same summary state? The labels are A, B, C. For query A, use attention weights proportional to "}<InlineMath>{"[4,1,1]"}</InlineMath>{". Compute both attention reads and explain what this proves."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Unroll the numerator to "}<InlineMath>{"\\frac14v_1+\\frac12v_2+v_3"}</InlineMath>{". The normalizer is the same for equal sequence lengths."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Both numerators are 4.25 and both normalizers are 1.75, giving 17/7. Attention gives "}<InlineMath>{"(4+4+2)/6=5/3"}</InlineMath>{" for the first sequence and "}<InlineMath>{"(20+2+2)/6=4"}</InlineMath>{" for the second. This summary cannot distinguish these histories; separate keyed values can. Neither attention result is guaranteed to equal its label-A value, and the example does not prove that all recurrent states have this collision."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"2. A different cache budget"}</H3>

<Prose>{"A model has 24 layers, six using full attention. Batch size is two, with four KV heads of width 64 and two-byte K/V storage. At 8,192 tokens, calculate K/V bytes. Then double only the sequence length. Which other model information would be needed to calculate total request state?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Multiply batch, attention layers, tokens, two banks, KV heads, head width and bytes per scalar."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The count is "}<InlineMath>{"2\\cdot6\\cdot8192\\cdot2\\cdot4\\cdot64\\cdot2=100,663,296"}</InlineMath>{" bytes, or 96 MiB. Doubling context gives 192 MiB. Recurrent-layer count, expanded/state dimensions, convolution history layout and their dtypes are needed for the remaining persistent state. Allocator overhead, weights and temporary memory are additional concerns."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"3. Repair the residual"}</H3>

<Prose>{"An implementation uses "}<InlineMath>{"x_{\\mathrm{next}}=x+\\mathrm{FFN}(\\mathrm{Norm}(x+\\mathrm{Mix}(\\mathrm{Norm}(x))))"}</InlineMath>{". Explain the missing path. In a scalar example with "}<InlineMath>{"x=2"}</InlineMath>{", mixer output 3 and FFN output 4, compare it with the intended two-residual layer."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Name the intermediate value after the mixer addition before writing the second addition."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The first intermediate value is "}<InlineMath>{"u=2+3=5"}</InlineMath>{". The intended result is "}<InlineMath>{"u+4=9"}</InlineMath>{". The displayed implementation returns "}<InlineMath>{"2+4=6"}</InlineMath>{", so the mixer contribution lacks its direct residual path into the final output. The FFN still depends on it, but that is a different function."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"4. Where did the missing probability mass go?"}</H3>

<Prose>{"Four router probabilities are "}<InlineMath>{"[0.4,0.3,0.2,0.1]"}</InlineMath>{". Top-2 outputs are "}<InlineMath>{"[5,0]"}</InlineMath>{" and "}<InlineMath>{"[0,5]"}</InlineMath>{". Compute the retained-probability mixture and the renormalized mixture. If only the third expert's output changes, which mixture changes?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"The selected mass is 0.7. Selection and score computation are held fixed."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Retained mixing gives "}<InlineMath>{"[2,1.5]"}</InlineMath>{". Renormalized mixing gives "}<InlineMath>{"[20/7,15/7]"}</InlineMath>{". Neither changes when only an unselected expert's output changes, since that output is not evaluated in the mixture. Changing its router score is different and can affect the retained weights even before selection changes."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"5. A correct final label hides a cache bug"}</H3>

<Prose>{"The model returns digit zero both before and after its convolution history is cleared mid-request. A teammate concludes that convolution history can be discarded. Give a concrete verification strategy and a null case where clearing state really has no effect."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Compare numeric outputs at every position under the same weights and inputs, not just an argmax."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Compare full-sequence logits with correctly carried streaming logits, then compare the damaged branch. Check more than one input and inspect the point immediately after the boundary. Intermediate differences or probability changes refute equivalence even when the final class stays zero. Clearing an already-empty cache before the first token is a true null; clearing after the last token cannot alter already-produced outputs. These nulls do not justify clearing a nonempty midstream cache."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"6. Choose without looking at the assessment answers"}</H3>

<Prose>{"Two candidate models have validation cross-entropies 0.12 and 0.15; their assessment error counts are 50 and 30. Your declared selection rule was lowest validation cross-entropy. Which candidate is selected, and what do you do with the second candidate's better assessment count?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Separate an honest report from a new selection decision."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Select the first candidate under the declared rule. Report both assessment results as part of the planned comparison, including the reversal. Do not relabel the second as the validation-selected winner. If the reversal motivates a new selection rule or training experiment, develop it using development evidence and evaluate the resulting decision with an independent assessment protocol."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"7. Shared weights do not guarantee shared keys"}</H3>

<Prose>{"A shared key projection is the identity. It is called at two depths with vectors "}<InlineMath>{"[1,0]"}</InlineMath>{" and "}<InlineMath>{"[0,1]"}</InlineMath>{" for the same token. Can the two invocations use the same stored key without changing either computation? What extra design would make cache sharing legitimate?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Apply the projection before arguing from parameter identity."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The keys are "}<InlineMath>{"[1,0]"}</InlineMath>{" and "}<InlineMath>{"[0,1]"}</InlineMath>{", so replacing one with the other changes its attention scores. Explicit architectural sharing could define both layers to consume a key bank generated by a designated layer, with that rule used during training and inference. That is a different contract from merely tying projection weights."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"8. Active weights and stored weights"}</H3>

<Prose>{"A toy network has eight FFN positions. Four are dense; four have four experts and select two. Each SwiGLU expert has model width 16 and intermediate width 32. Calculate total FFN parameters, active FFN parameters for one token, and the corresponding all-dense total."}</Prose>

<details><summary>Hint</summary>

<Prose>{"One expert has three matrices, each with "}<InlineMath>{"16\\cdot32"}</InlineMath>{" entries."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"One expert has 1,536 parameters. Stored FFN parameters are "}<InlineMath>{"(4+4\\cdot4)1536=30,720"}</InlineMath>{". Active FFN parameters are "}<InlineMath>{"(4+4\\cdot2)1536=18,432"}</InlineMath>{". Eight dense FFNs have 12,288. Sparse selection increases the parameter pool while limiting, rather than eliminating, extra work. Router and mixer parameters are excluded."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"9. Window the attention"}</H3>

<Prose>{"Replace every full-attention layer by a window of 512 positions. A query asks for an identifier introduced 10,000 positions earlier. Explain what information routes remain and why the cache can be bounded without proving the answer will be correct."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Direct access to a key is different from information carried forward in hidden representations."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The old identifier's original key lies outside every individual attention window. Its information could persist through recurrent state or be copied into later representations and relayed through layers. Those routes may or may not preserve enough detail. With fixed width and state size, each layer's window cache and recurrent buffers have bounded size, but that storage bound is not a guarantee of arbitrary exact recall."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"10. Design a useful hybrid experiment"}</H3>

<Prose>{"You want to answer questions about a laboratory notebook. Design two interventions that distinguish identifier retrieval from combining evidence across a sequence, and name the measurements required to decide whether a hybrid is useful."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Change one explanatory factor at a time. Include questions whose answers are absent."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"For retrieval, change one experiment's identifier or final parameter value while preserving the surrounding narrative and add near-matching distractor identifiers. For sequence integration, reorder a superseded setting and its correction, or move the relevant evidence across several separated entries while holding identifier vocabulary fixed. Score correct answers, evidence citations and appropriate no-evidence responses. Measure TTFT, decode time, total latency, peak memory and concurrency at a recorded model/software/hardware configuration. Compare with a simpler model or structured retrieval baseline under the same task and resource constraints."}</Prose>

</details></div></section>

<section className="lesson-ending lesson-ending--resources" data-lesson-ending="resources"><H2>{"12. References and other ways to learn"}</H2>

<ul><li>{""}<strong>{"Architecture and evidence:"}</strong>{" "}<a href={"https://arxiv.org/html/2403.19887v1"}>{"Lieber et al., Jamba"}</a>{". Start with §2 and the actual release shape in §3; then read the ratio, format-following and normalization investigations in §6. The authors distinguish measured behavior from their induction-head hypothesis."}</li><li>{""}<strong>{"Scale and serving:"}</strong>{" "}<a href={"https://arxiv.org/html/2408.12570v1"}>{"Jamba-1.5 technical report"}</a>{". §§2–3 explain the larger configuration, ExpertsInt8 and activation-range issue; §§5–6 explain training stages and evaluation. Read the hardware and output-length conditions beside throughput figures."}</li><li>{""}<strong>{"Implementation reference:"}</strong>{" "}<a href={"https://huggingface.co/docs/transformers/model_doc/jamba"}>{"Hugging Face Jamba documentation"}</a>{" and "}<a href={"https://github.com/huggingface/transformers/blob/main/src/transformers/models/jamba/modeling_jamba.py"}>{"model source"}</a>{". Follow the two residual additions, normalized timestep/B/C features, selected router weights and distinct cache operations. Names and kernel APIs can change between revisions."}</li><li>{""}<strong>{"Data and independent reproduction:"}</strong>{" "}<a href={"https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits"}>{"UCI PenDigits"}</a>{", by E. Alpaydin and F. Alimoglu, "}<a href={"https://doi.org/10.24432/C5MG6K"}>{"DOI"}</a>{". The original names file explains spatial resampling and the writer-disjoint official assessment set. The accompanying files retain attribution and the CC BY 4.0 source license."}</li><li>{""}<strong>{"A guided video course:"}</strong>{" "}<a href={"https://www.deeplearning.ai/courses/build-long-context-ai-apps-with-jamba/"}>{"Build Long-Context AI Apps with Jamba"}</a>{", DeepLearning.AI with AI21, taught by Chen Wang and Chen Almagor. The verified course listing describes nine video lessons, architecture, document prompting, tool calling and context/RAG applications. Its listing was reviewed; the videos and notebooks were not watched or executed for this lesson. Use it for an application-oriented second explanation; access and hosted APIs may require an account."}</li><li>{""}<strong>{"Alternative design explanations:"}</strong>{" "}<a href={"https://www.zyphra.com/our-work/the-zyphra-training-cookbook"}>{"Zyphra's training cookbook"}</a>{" explains the rationale for shared blocks; pair it with "}<a href={"https://arxiv.org/html/2405.16712v1"}>{"Zamba"}</a>{" and the "}<a href={"https://huggingface.co/Zyphra/Zamba2-7B"}>{"Zamba2 card"}</a>{". "}<a href={"https://arxiv.org/html/2406.07522v1"}>{"Samba"}</a>{" and "}<a href={"https://arxiv.org/html/2411.13676v1"}>{"Hymba"}</a>{" expose different choices of locality and within-layer fusion."}</li><li>{""}<strong>{"Serving-state design:"}</strong>{" "}<a href={"https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/"}>{"vLLM hybrid KV cache manager"}</a>{". Useful after the cache investigation; it shows why allocator groups, padding and prefix-reuse rules cannot all be copied from a homogeneous attention stack."}</li></ul>

<Prose>{"The next topic in this module is "}<a href={"/learn/path/full-curriculum/titans-multi-memory-architecture?module=deep-learning-fundamentals"}>{"Titans: Multi-Memory Architecture"}</a>{". Carry forward the questions learned here: what each memory stores, what updates it, how a read is performed, and what must be preserved when processing the next part of a sequence."}</Prose>
<JambaProgram file="hybrid_mechanisms.py"/><JambaDownloads/></section>
</div>};
