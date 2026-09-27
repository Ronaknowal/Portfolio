// Generated from the complete prepared manuscript by scripts/generate-grouped-query-lesson.mjs.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements';
import { GqaCacheTimeline, GqaWiring, GqaReadLab, GqaMaskLab, GqaBudgetLab, GqaConversionLab, GqaForecastLab, GqaGradientFigure, GqaProgram, GqaTrainingHistory } from '../../components/lesson-labs/GroupedQueryLabs';
import { GqaCompactWriteDiagram, GqaOwnershipDiagram, GqaDistributedDiagram } from '../../components/lesson-labs/GroupedQueryDiagrams.jsx';
import '../../components/lesson-labs/neural-lesson-neutral.css';
const lesson = { title: 'Grouped-Query Attention (GQA) & Multi-Query Attention (MQA)', readTime: '~65 min read + 90 min practice', content: () => <div className="gqa-lesson neural-lesson neural-lesson-neutral">
<Prose opening="exploration">{""}<strong>{"Explore as you read."}</strong>{" Edit Q/K/V, query-to-KV grouping, cache dimensions, offset masks and supported causal input prefixes. Show each reader, shared K/V record, weighted sum, exact byte/MAC budgets and compact cache outputs immediately. Compare equal-head versus unequal-head regrouping. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose grouping by memory and functional tradeoffs, keeping payload arithmetic separate from measured latency and model quality."}</Prose>

<Prose>{"A model predicting the next word repeatedly reads what it has already seen. Several attention heads can ask different questions about that history. Must each head keep its own separate description of every earlier token?"}</Prose>

<Prose>{"Grouped-query attention lets several query heads read the same keys and values. Multi-query attention shares one key/value head across all query heads. The important distinction is between "}<strong>{"how many different reads we perform"}</strong>{" and "}<strong>{"how many different representations we store"}</strong>{". Sharing the stored representation can reduce the growing inference cache while retaining several distinct attention distributions."}</Prose>

<Prose>{"The "}<a href={"/learn/path/full-curriculum/positional-encodings-sinusoidal-learned-rope-alibi?module=deep-learning-fundamentals"}>{"previous lesson on positional encodings"}</a>{" explained where positions enter those representations. Here we trace the actual grouped computation, build a compact causal cache and convert a small trained model. Our real example forecasts the next point of a recorded hand movement; it makes the known-prefix versus generated-future distinction visible without downloading a language model."}</Prose>

<Prose opening="route">{""}<strong>{"First pass:"}</strong>{" follow §§1–6 and exercises 1–5. You will be able to calculate a grouped head's output, implement correct caching, explain the storage savings and assess a conversion experiment. The deeper route in §7 develops gradients, representation constraints and serving tradeoffs; exercises 6–9 extend those ideas. The diagrams and short arithmetic examples belong alongside their explanations, not in a separate optional gallery."}</Prose>

<H2>{"1. The growing memory behind one prediction"}</H2>

<H3>{"Read the history without recomputing it"}</H3>

<Prose>{"Recall "}<a href={"/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals"}>{"Self-Attention"}</a>{": a query is compared with keys, the resulting scores are normalized into weights, and those weights mix values. In a causal decoder, the current position may read itself and preceding positions. Later inputs are unavailable."}</Prose>

<Prose>{"Suppose a decoder has already processed positions 0–31. It has computed their keys and values at every layer. When position 32 arrives, earlier causal hidden states do not need to change under a fixed model and position rule: none depended on the new future point. Store those past K/V tensors, project the new input once, append its K/V, and use the new query to read the enlarged history. This stored state is the "}<strong>{"KV cache"}</strong>{"."}</Prose>

<Prose>{"Why do we generally not cache old queries? The next output uses a new query. The keys and values are the old information it reads. Old queries are no longer needed for this ordinary incremental attention step. Other algorithms can store additional state, but that does not alter this basic dependency."}</Prose>

<GqaCacheTimeline />

<H3>{"Prefill and decode are different workloads"}</H3>

<Prose>{"The prompt or observed prefix is already known. Its positions can be processed together with a causal mask; this is "}<strong>{"prefill"}</strong>{". During generation, a newly predicted word or point becomes the input for a later prediction, so there is a sequential dependency. This is "}<strong>{"decode"}</strong>{". A real decoder can batch independent requests and can verify proposed chunks, but it must preserve the relevant information dependencies."}</Prose>

<Prose>{"With many known queries, the same K/V block can contribute to many calculations while resident near the arithmetic units. With one new query per request, repeatedly moving the growing history from device memory can be costly compared with the work done on it. This is the bandwidth motivation behind "}<a href={"https://arxiv.org/pdf/1911.02150"}>{"Shazeer's multi-query attention paper"}</a>{". It is not a claim that KV reads dominate every model, batch size, sequence length or device. Weight reads, feedforward arithmetic, communication and scheduling also matter."}</Prose>

<Prose>{"Sharing K/V heads addresses the "}<strong>{"head axis"}</strong>{" of this history. It does not by itself remove old token positions, compress the sequence into a fixed recurrent state or change the causal mask."}</Prose>

<H2>{"2. Several questions, fewer stored descriptions"}</H2>

<H3>{"Use names that cannot swap meanings"}</H3>

<Prose>{"We will use:"}</Prose>

<NeuralTable caption={"Use names that cannot swap meanings"} headers={[<>{"Symbol"}</>,<>{"Meaning"}</>]} rows={[[<>{""}<InlineMath>{"H_q"}</InlineMath>{""}</>,<>{"Number of query heads"}</>],[<>{""}<InlineMath>{"H_{kv}"}</InlineMath>{""}</>,<>{"Number of distinct key heads and value heads"}</>],[<>{""}<InlineMath>{"R=H_q/H_{kv}"}</InlineMath>{""}</>,<>{"Number of query heads sharing each K/V head"}</>],[<>{""}<InlineMath>{"d_k"}</InlineMath>{""}</>,<>{"Coordinate width of one query/key head"}</>],[<>{""}<InlineMath>{"d_v"}</InlineMath>{""}</>,<>{"Coordinate width of one value head"}</>],[<>{""}<InlineMath>{"D"}</InlineMath>{""}</>,<>{"Width of the model's input/output rows"}</>]]} />

<Prose>{"The common equal-group construction requires positive head counts with "}<InlineMath>{"H_q"}</InlineMath>{" divisible by "}<InlineMath>{"H_{kv}"}</InlineMath>{". It also uses the same K-head and V-head count. Query and key widths must match for their dot product; value width can differ. A frequent architecture chooses "}<InlineMath>{"d_k=d_v=D/H_q"}</InlineMath>{", but the operator itself does not require this equality."}</Prose>

<Prose>{"For eight query heads:"}</Prose>

<NeuralTable caption={"Use names that cannot swap meanings"} headers={[<>{"Mechanism"}</>,<>{""}<InlineMath>{"H_q"}</InlineMath>{""}</>,<>{""}<InlineMath>{"H_{kv}"}</InlineMath>{""}</>,<>{""}<InlineMath>{"R"}</InlineMath>{""}</>,<>{"Query-head to KV-head mapping"}</>]} rows={[[<>{"Multi-head attention, MHA"}</>,<>{"8"}</>,<>{"8"}</>,<>{"1"}</>,<>{"0,1,2,3,4,5,6,7"}</>],[<>{"Grouped-query attention, GQA"}</>,<>{"8"}</>,<>{"4"}</>,<>{"2"}</>,<>{"0,0,1,1,2,2,3,3"}</>],[<>{"More sharing"}</>,<>{"8"}</>,<>{"2"}</>,<>{"4"}</>,<>{"0,0,0,0,1,1,1,1"}</>],[<>{"Multi-query attention, MQA"}</>,<>{"8"}</>,<>{"1"}</>,<>{"8"}</>,<>{"0,0,0,0,0,0,0,0"}</>]]} />

<Prose>{"In "}<a href={"https://arxiv.org/html/2305.13245v3#S2.SS2"}>{"Ainslie et al.'s notation"}</a>{", “GQA-8” means "}<strong>{"eight KV groups"}</strong>{", not eight queries per group. A model with 32 queries and 8 KV heads has group size 4. A model with 64 queries and 8 KV heads has group size 8. Both have eight KV groups. Keeping these numbers separate prevents several apparent contradictions in model descriptions."}</Prose>

<H3>{"A shared key does not mean a shared question"}</H3>

<Prose>{"Imagine two readers consulting the same indexed records. One asks about the beginning of a movement; another asks about its most recent direction. They can assign different importance to the same records. The analogy describes shared information, not a guarantee that a particular head learns a human-named role."}</Prose>

<Prose>{"In equations, the keys and values are shared, but "}<InlineMath>{"q_0"}</InlineMath>{" and "}<InlineMath>{"q_1"}</InlineMath>{" can differ. Their dot products, softmax weights and mixed outputs can therefore differ. MQA with eight query heads still produces eight head outputs. It is not single-head attention with a new name."}</Prose>

<GqaWiring />

<Prose>{"Contiguous groups are a layout convention. An interleaved assignment can define a valid architecture if training, weights and inference consistently use it. Swapping from contiguous to interleaved routing while retaining a checkpoint's unchanged parameters generally changes its function. That is the bug—not a theorem that every noncontiguous grouping is intrinsically inferior."}</Prose>

<H2>{"3. Calculate the grouped operation"}</H2>

<H3>{"From projections to head outputs"}</H3>

<Prose>{"Let the query inputs have "}<InlineMath>{"T"}</InlineMath>{" rows, memory inputs have "}<InlineMath>{"S"}</InlineMath>{" rows and batch size be "}<InlineMath>{"B"}</InlineMath>{". In full self-attention these can be the same input; in an incremental step "}<InlineMath>{"T"}</InlineMath>{" may be 1 while "}<InlineMath>{"S"}</InlineMath>{" includes the prefix. After learned projections and head reshaping:"}</Prose>

<div className="neural-equation"><MathBlock>{"Q\\in\\mathbb R^{B\\times H_q\\times T\\times d_k},\\quad\nK\\in\\mathbb R^{B\\times H_{kv}\\times S\\times d_k},\\quad\nV\\in\\mathbb R^{B\\times H_{kv}\\times S\\times d_v}."}</MathBlock></div>

<Prose>{"Under the contiguous convention, query head "}<InlineMath>{"h"}</InlineMath>{" reads memory head "}<InlineMath>{"g(h)=\\lfloor h/R\\rfloor"}</InlineMath>{". Its score and output are"}</Prose>

<div className="neural-equation"><MathBlock>{"s_{h,t,j}=\\frac{q_{h,t}^{\\top}k_{g(h),j}}{\\sqrt{d_k}}+b_{h,t,j},\\quad\na_{h,t,:}=\\operatorname{softmax}_{\\text{legal keys}}(s_{h,t,:}),\\quad\no_{h,t}=\\sum_j a_{h,t,j}v_{g(h),j}."}</MathBlock></div>

<Prose>{"Here "}<InlineMath>{"b"}</InlineMath>{" may contain an appropriate position bias. An illegal key is excluded, conventionally by a score of negative infinity before softmax. Setting an illegal score to zero does not exclude it: its exponential would be 1."}</Prose>

<Prose>{"Concatenate all "}<InlineMath>{"H_q"}</InlineMath>{" output heads, producing width "}<InlineMath>{"H_qd_v"}</InlineMath>{", then apply an output map back to width "}<InlineMath>{"D"}</InlineMath>{". This is followed by the residual and feedforward processing from "}<a href={"/learn/path/full-curriculum/transformer-block-architecture?module=deep-learning-fundamentals"}>{"Transformer Block Architecture"}</a>{". Sharing K/V changes the K/V projection shapes inside attention; it leaves the external row width compatible with the rest of the block."}</Prose>

<H3>{"A four-query, two-memory example"}</H3>

<Prose>{"Use one query position, three legal memory positions and "}<InlineMath>{"d_k=d_v=2"}</InlineMath>{". These are deliberately chosen arithmetic inputs, not learned language-model features."}</Prose>

<Prose>{"The four queries are"}</Prose>

<div className="neural-equation"><MathBlock>{"q_0=\\sqrt2[1,0],\\quad q_1=\\sqrt2[0,1],\\quad\nq_2=\\sqrt2[-1,0],\\quad q_3=\\sqrt2[0,-1]."}</MathBlock></div>

<Prose>{"Each group holds three keys and values:"}</Prose>

<NeuralTable caption={"A four-query, two-memory example"} headers={[<>{"Position"}</>,<>{"Group 0 key"}</>,<>{"Group 0 value"}</>,<>{"Group 1 key"}</>,<>{"Group 1 value"}</>]} rows={[[<>{"0"}</>,<>{"[1,0]"}</>,<>{"[2,0]"}</>,<>{"[1,1]"}</>,<>{"[1,3]"}</>],[<>{"1"}</>,<>{"[0,1]"}</>,<>{"[0,4]"}</>,<>{"[−1,0]"}</>,<>{"[−1,2]"}</>],[<>{"2"}</>,<>{"[1,1]"}</>,<>{"[2,2]"}</>,<>{"[0,−1]"}</>,<>{"[3,0]"}</>]]} />

<Prose>{"Queries 0 and 1 read group 0; queries 2 and 3 read group 1. For query 0, dividing by "}<InlineMath>{"\\sqrt2"}</InlineMath>{" cancels the chosen query factor, giving scores "}<code>{"[1,0,1]"}</code>{". Their exponentials are "}<code>{"[e,1,e]"}</code>{", so the weights are approximately "}<code>{"[0.422319,0.155362,0.422319]"}</code>{". The output is"}</Prose>

<div className="neural-equation"><MathBlock>{"0.422319[2,0]+0.155362[0,4]+0.422319[2,2]\n\\approx[1.689275,1.466087]."}</MathBlock></div>

<Prose>{"Performing the same calculation for the other queries gives:"}</Prose>

<NeuralTable caption={"A four-query, two-memory example"} headers={[<>{"Query head"}</>,<>{"KV group"}</>,<>{"Scaled scores"}</>,<>{"Weights over positions 0,1,2"}</>,<>{"Mixed output"}</>]} rows={[[<>{"0"}</>,<>{"0"}</>,<>{"[1,0,1]"}</>,<>{"[0.422319,0.155362,0.422319]"}</>,<>{"[1.689275,1.466087]"}</>],[<>{"1"}</>,<>{"0"}</>,<>{"[0,1,1]"}</>,<>{"[0.155362,0.422319,0.422319]"}</>,<>{"[1.155362,2.533913]"}</>],[<>{"2"}</>,<>{"1"}</>,<>{"[−1,1,0]"}</>,<>{"[0.090031,0.665241,0.244728]"}</>,<>{"[0.158975,1.600574]"}</>],[<>{"3"}</>,<>{"1"}</>,<>{"[−1,0,1]"}</>,<>{"[0.090031,0.244728,0.665241]"}</>,<>{"[1.841025,0.759549]"}</>]]} />

<Prose>{"The two readers of group 0 plainly have different answers. Their shared memory did not force their attention weights to coincide."}</Prose>

<H3>{"Edit a head and inspect which outputs change"}</H3>

<Prose>{"Change the first value of group 0 from "}<code>{"[2,0]"}</code>{" to "}<code>{"[3,−1]"}</code>{". No score changes because scores use Q and K. The two group-0 outputs change by their respective weight on that position times "}<code>{"[1,−1]"}</code>{". Heads 2 and 3 are unchanged because they read another group."}</Prose>

<Prose>{"Now instead change group 0's first key from "}<code>{"[1,0]"}</code>{" to "}<code>{"[2,0]"}</code>{". Query 0's first score increases. Query 1's score does "}<strong>{"not"}</strong>{" change in this special fixture: its query has zero x component. A shared key edit can influence every reader in its group, but it need not do so for every particular query. This controlled null is more informative than an animation that always lights up all arrows as “affected.”"}</Prose>

<GqaReadLab />

<H3>{"Repeat is one implementation, not the definition"}</H3>

<Prose>{"An easy reference implementation repeats each KV head "}<InlineMath>{"R"}</InlineMath>{" times and calls ordinary multi-head attention. This gives the correct function, but explicit "}<code>{"repeat_interleave"}</code>{" allocates repeated tensors. Its gradients are valid because backward sums the contributions from copies; valid differentiation does not make the forward a zero-cost view."}</Prose>

<Prose>{"We can instead reshape queries as "}<code>{"[B,Hkv,R,T,dk]"}</code>{" and keep K/V compact. Compute scores with"}</Prose>

<CodeBlock language={"python"}>{"scores = torch.einsum(\"bgrtd,bgud->bgrtu\", grouped_query, keys) / math.sqrt(dk)\nweights = scores.softmax(dim=-1)\noutputs = torch.einsum(\"bgrtu,bgud->bgrtd\", weights, values)"}</CodeBlock>

<Prose>{"The letters name axes: "}<code>{"g"}</code>{" is a KV group, "}<code>{"r"}</code>{" a reader within it, "}<code>{"t"}</code>{" a query position and "}<code>{"u"}</code>{" a memory position. Summing over "}<code>{"d"}</code>{" makes scores; summing over "}<code>{"u"}</code>{" mixes values. A mask belongs before softmax. The full executable program below includes it."}</Prose>

<Prose>{"This eager grouped form avoids an explicit repeated K/V array. It still forms one score distribution per "}<strong>{"query head"}</strong>{", so its score tensor has "}<InlineMath>{"BH_qTS"}</InlineMath>{" entries. A fused kernel can tile this work and reuse K/V within its execution strategy. Do not infer a particular physical memory-traffic count merely from a high-level "}<code>{"einsum"}</code>{"."}</Prose>

<H2>{"4. Build the cache and count the savings honestly"}</H2>

<H3>{"Store compact K/V, retain the position contract"}</H3>

<Prose>{"At a new position, project "}<InlineMath>{"H_q"}</InlineMath>{" queries and only "}<InlineMath>{"H_{kv}"}</InlineMath>{" keys and values. If using standard RoPE, rotate each query and each unique key at its proper logical position, then append the compact rotated keys and unrotated values to the cache. Read them using the group mapping."}</Prose>

<Prose>{"If all copies receive the same position and frequency rule, “rotate then repeat” and “repeat then rotate” are mathematically equivalent. Rotating the unique keys avoids redundant work. Cache correctness depends on retaining a consistent rotated/unrotated convention and logical IDs; the mere ordering of two equivalent operations does not force an expanded cache or a position bug."}</Prose>

<Prose>{"A one-position query tensor can correspond to logical position 32. The legal keys are positions 0–32, not only position 0. More generally, a new chunk at positions 3 and 4 against keys 0–4 needs"}</Prose>

<CodeBlock language={"text"}>{"query 3: 1 1 1 1 0\nquery 4: 1 1 1 1 1"}</CodeBlock>

<Prose>{"This is a logical-position relation. A generic upper-left triangle on a 2-by-5 tensor would instead permit only the first one and first two keys. Both arrays have valid shapes; only one describes this cached computation."}</Prose>

<><GqaCompactWriteDiagram /><GqaMaskLab /></>

<H3>{"Bytes come from actual tensor dimensions"}</H3>

<Prose>{"For a uniform unquantized cache with batch size "}<InlineMath>{"B"}</InlineMath>{", "}<InlineMath>{"N"}</InlineMath>{" layers, prefix length "}<InlineMath>{"L"}</InlineMath>{", equal K/V head count "}<InlineMath>{"H_{kv}"}</InlineMath>{" and bytes per stored number "}<InlineMath>{"s"}</InlineMath>{", the K/V tensor payload is"}</Prose>

<div className="neural-equation"><MathBlock>{"C=B N L H_{kv}(d_k+d_v)s."}</MathBlock></div>

<Prose>{"When "}<InlineMath>{"d_k=d_v=d_h"}</InlineMath>{", this becomes "}<InlineMath>{"2BNLH_{kv}d_hs"}</InlineMath>{". One factor 2 means “key plus value”; another factor may come from using two-byte FP16/BF16 numbers. Do not merge them and accidentally halve the answer."}</Prose>

<Prose>{"For an illustrative architecture with 80 layers, 64 query heads, width 128 per K/V head and two-byte cache entries:"}</Prose>

<NeuralTable caption={"Bytes come from actual tensor dimensions"} headers={[<>{"Distinct KV heads"}</>,<>{"Cache bytes per token, one sequence"}</>,<>{"Payload at 32,768 tokens"}</>,<>{"Same payload in decimal GB"}</>]} rows={[[<>{"64, MHA"}</>,<>{"2,621,440"}</>,<>{"80 GiB"}</>,<>{"85.899 GB"}</>],[<>{"8, GQA"}</>,<>{"327,680"}</>,<>{"10 GiB"}</>,<>{"10.737 GB"}</>],[<>{"1, MQA"}</>,<>{"40,960"}</>,<>{"1.25 GiB"}</>,<>{"1.342 GB"}</>]]} />

<Prose>{"A GiB is "}<InlineMath>{"2^{30}"}</InlineMath>{" bytes; a decimal GB is "}<InlineMath>{"10^9"}</InlineMath>{" bytes. These are calculated tensor sizes for the stated configuration. They do not claim that a named checkpoint was trained for this context length or that the complete model fits a particular device."}</Prose>

<Prose>{"The reduction from 64 KV heads to 8 is eightfold; from 8 to 1 is another eightfold; from 64 to 1 is sixty-fourfold. Group size, head count and the chosen comparison determine the ratio."}</Prose>

<Prose>{"For variable request lengths and different attention layer types, sum the actual per-request, per-layer terms. An allocated cache may reserve maximum lengths or rounded pages instead of exactly the occupied tokens. Quantized caches also have scales, packing and other metadata; four-bit values do not imply that every stored quantity occupies exactly half a byte. Prefix sharing and distributed replication further distinguish logical tensor payload from physical allocation."}</Prose>

<H3>{"Parameter arithmetic changes too"}</H3>

<Prose>{"Without biases, projections have parameter counts"}</Prose>

<div className="neural-equation"><MathBlock>{"P_Q=DH_qd_k,\\quad P_K=DH_{kv}d_k,\\quad\nP_V=DH_{kv}d_v,\\quad P_O=DH_qd_v."}</MathBlock></div>

<Prose>{"Their sum is "}<InlineMath>{"D(H_q+H_{kv})(d_k+d_v)"}</InlineMath>{". For the common "}<InlineMath>{"d_k=d_v=D/H_q"}</InlineMath>{" case:"}</Prose>

<div className="neural-equation"><MathBlock>{"P_{\\rm attention}=2D^2\\left(1+\\frac{H_{kv}}{H_q}\\right)."}</MathBlock></div>

<Prose>{"With "}<InlineMath>{"D=512,H_q=8"}</InlineMath>{", MHA has 1,048,576 attention-projection parameters; two-KV-head GQA has 655,360; MQA has 589,824. Biases, norms, feedforward layers and embeddings add their own parameters. The percentage saved in the "}<strong>{"whole model"}</strong>{" depends on that full architecture. It is not always 5%, and optimizer-state memory can also decrease when trainable parameter count decreases."}</Prose>

<H3>{"Which arithmetic remains?"}</H3>

<Prose>{"Ignoring small overheads and counting a multiply-add as two operations, one attention application needs approximately"}</Prose>

<div className="neural-equation"><MathBlock>{"2BH_qTSd_k\\quad\\text{for QK scores},\\qquad\n2BH_qTSd_v\\quad\\text{for mixing values}."}</MathBlock></div>

<Prose>{"These terms retain "}<InlineMath>{"H_q"}</InlineMath>{", even when K/V are shared. Different queries still form different weighted sums. The K/V "}<strong>{"projection"}</strong>{" arithmetic does decrease with "}<InlineMath>{"H_{kv}"}</InlineMath>{". Thus “all FLOPs stay identical” is too broad; “the dense score and value-mixing arithmetic is unchanged at fixed query heads, widths and lengths” is the useful precise statement."}</Prose>

<Prose>{"For one decode query with equal widths, these attention operations total about "}<InlineMath>{"4BH_qLd_h"}</InlineMath>{". An idealized read-once compact K/V payload is "}<InlineMath>{"2BH_{kv}Ld_hs"}</InlineMath>{", suggesting arithmetic per byte of "}<InlineMath>{"2R/s"}</InlineMath>{". With two-byte elements this is "}<InlineMath>{"R"}</InlineMath>{" operations per byte. This model assumes effective reuse and omits weights, writes, cache hierarchy and other work; it explains why sharing can help a bandwidth-limited kernel without predicting its measured speed."}</Prose>

<Prose>{"If 60% of a step's time were reducible by a factor of 8 and everything else stayed fixed, the total speedup would be"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{1}{0.4+0.6/8}\\approx2.105,"}</MathBlock></div>

<Prose>{"not 8. This is an illustrative Amdahl calculation, not an observed decoder timing. Real group counts may also change kernel occupancy, parallelism, batching and communication. Measure the workload before turning a storage ratio into a latency claim."}</Prose>

<GqaBudgetLab />

<H2>{"5. Implement the mechanism and verify a real API contract"}</H2>

<H3>{"A complete transparent NumPy reference"}</H3>

<Prose>{"This program loops over query heads so the assignment is easy to inspect. It does not repeat the stored K/V array. Python and NumPy are sufficient; no trained weights or GPU are needed. The displayed arrays are the hand example from §3."}</Prose>

<CodeBlock language={"python"}>{"import math\nimport numpy as np\n\n\ndef grouped_attention(query, keys, values, query_positions, key_positions,\n                      mapping=None, causal=True):\n    # Q: [B,Hq,T,dk], K: [B,Hkv,S,dk], V: [B,Hkv,S,dv].\n    batch, query_heads, query_length, key_width = query.shape\n    kv_heads = keys.shape[1]\n    if query_heads % kv_heads or keys.shape[:3] != values.shape[:3]:\n        raise ValueError(\"Use equal-size groups and matching K/V head/slot counts.\")\n    if keys.shape[0] != batch or keys.shape[-1] != key_width:\n        raise ValueError(\"Batch and Q/K coordinate widths must agree.\")\n    if mapping is None:\n        mapping = np.arange(query_heads) // (query_heads // kv_heads)\n    mapping = np.asarray(mapping)\n    if (mapping.shape != (query_heads,) or np.any(mapping < 0)\n            or np.any(mapping >= kv_heads)):\n        raise ValueError(\"Each query head must name a valid KV head.\")\n    legal = np.ones((query_length, keys.shape[2]), dtype=bool)\n    if causal:\n        legal = np.asarray(key_positions)[None, :] <= np.asarray(query_positions)[:, None]\n    if not legal.any(-1).all():\n        raise ValueError(\"A query has no legal key.\")\n    outputs, weights = [], []\n    for head, memory_head in enumerate(mapping):\n        scores = query[:, head] @ keys[:, memory_head].swapaxes(-1, -2)\n        scores /= math.sqrt(key_width)\n        scores = np.where(legal, scores, -np.inf)\n        probabilities = np.exp(scores - scores.max(-1, keepdims=True))\n        probabilities /= probabilities.sum(-1, keepdims=True)\n        outputs.append(probabilities @ values[:, memory_head])\n        weights.append(probabilities)\n    return np.stack(outputs, 1), np.stack(weights, 1)\n\n\nq = np.array([[1,0], [0,1], [-1,0], [0,-1]], dtype=float)[None,:,None,:] * math.sqrt(2)\nk = np.array([[[1,0], [0,1], [1,1]], [[1,1], [-1,0], [0,-1]]], dtype=float)[None]\nv = np.array([[[2,0], [0,4], [2,2]], [[1,3], [-1,2], [3,0]]], dtype=float)[None]\noutput, weights = grouped_attention(q, k, v, [2], [0,1,2])\nprint(\"weights:\", np.round(weights[0,:,0], 6))\nprint(\"head outputs:\", np.round(output[0,:,0], 6))\n\n# The same function represented as MHA with tied/repeated K/V heads.\ntied, _ = grouped_attention(q, np.repeat(k, 2, axis=1),\n                           np.repeat(v, 2, axis=1), [2], [0,1,2])\nprint(\"tied MHA agrees:\", np.allclose(output, tied, atol=1e-12))\n\n# For one query at the last slot, a compact prefix plus the new entry agrees.\ncached_keys = np.concatenate((k[:,:,:2], k[:,:,2:]), axis=2)\ncached_values = np.concatenate((v[:,:,:2], v[:,:,2:]), axis=2)\ncached, _ = grouped_attention(q, cached_keys, cached_values, [2], [0,1,2])\nprint(\"compact cache agrees:\", np.allclose(output, cached, atol=1e-12))"}</CodeBlock>

<Prose>{"The head outputs are "}<code>{"[1.689275,1.466087]"}</code>{", "}<code>{"[1.155362,2.533913]"}</code>{", "}<code>{"[0.158975,1.600574]"}</code>{" and "}<code>{"[1.841025,0.759549]"}</code>{", and both checks print "}<code>{"True"}</code>{". The compact-cache check here verifies the "}<strong>{"projected-array assembly"}</strong>{". The full neural program in §6 additionally verifies that projecting and processing an observed sequence incrementally reproduces all full-pass causal predictions."}</Prose>

<Prose>{"This reference rejects an all-masked query instead of normalizing an empty set. A production kernel may specify another policy, such as a zero output, but the application must still distinguish “there is no legal evidence” from an ordinary attention distribution. Validate head counts, widths, masks and finite data at the actual input boundary."}</Prose>

<H3>{"Calling PyTorch SDPA is an explicit choice"}</H3>

<Prose>{"The "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html"}>{"PyTorch 2.14 SDPA contract"}</a>{" exposes "}<code>{"enable_gqa=True"}</code>{". It is false by default. Do not assume that fewer K/V heads automatically select the intended grouped operation in every API or backend. MQA can also happen to broadcast in some lower-level operations; that does not replace an explicit grouping contract."}</Prose>

<Prose>{"For existing tensors Q/K/V with the shapes above, the relevant call is:"}</Prose>

<CodeBlock language={"python"}>{"result = torch.nn.functional.scaled_dot_product_attention(\n    query, keys, values,\n    attn_mask=legal_mask,       # True means this key participates.\n    is_causal=False,           # The explicit mask already includes causal legality.\n    dropout_p=0.0,\n    enable_gqa=True,\n)"}</CodeBlock>

<Prose>{"This is a call-site fragment; "}<a href={"/learn-assets/grouped-query-attention-gqa-multi-query-attention-mqa/mechanism-calculations.py"}>{"the complete operator-check program"}</a>{" supplies the tensors and executes it. Its CPU float64 output agreed with the direct grouped calculation to "}<InlineMath>{"2.23\\times10^{-16}"}</InlineMath>{" in the recorded PyTorch 2.14.0 environment. That observation is not evidence about CUDA dispatch, throughput or every supported shape."}</Prose>

<GqaProgram file="mechanism-calculations.py" />

<Prose>{"Pay attention to two particularly easy API mismatches. SDPA's boolean mask uses "}<code>{"True"}</code>{" for "}<strong>{"allowed"}</strong>{", whereas "}<code>{"MultiheadAttention"}</code>{"'s key-padding mask uses "}<code>{"True"}</code>{" for "}<strong>{"excluded padding"}</strong>{". Also, SDPA's documented non-square "}<code>{"is_causal=True"}</code>{" aligns a triangle at the "}<strong>{"upper left"}</strong>{". A cached query at the last position needs the offset relation from §4. In our one-query/three-key fixture, the upper-left mask allows only the first key, returning its value in each group instead of the correct three-key mixture."}</Prose>

<Prose>{"The "}<a href={"https://github.com/Dao-AILab/flash-attention#how-to-use-flashattention"}>{"FlashAttention interface documentation"}</a>{" describes a bottom-right-aligned causal convention for its current unequal-length operation. These are different API contracts, despite similar parameter names. Compare each call with an explicit logical-position reference rather than transplanting a mask assumption between libraries. Backend support and fused-kernel constraints are version-specific; the recorded native checks do not require installing FlashAttention."}</Prose>

<Prose>{"SDPA also applies dropout according to the "}<code>{"dropout_p"}</code>{" argument. Setting a surrounding module to evaluation mode does not automatically override a nonzero argument passed to the functional call. Use zero for deterministic cached inference unless the application deliberately defines another behavior."}</Prose>

<Prose>{"The "}<a href={"https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py"}>{"maintained Llama source"}</a>{" provides a useful production reading exercise: inspect smaller K/V projections, native-head RoPE, compact cache update, then attention-backend dispatch. Its eager "}<code>{"repeat_kv"}</code>{" helper is a readable reference; it is not proof that an expand-plus-reshape path always avoids allocation. Inspect the actual storage and backend when that distinction matters."}</Prose>

<H3>{"Make sharing visible to the optimizer"}</H3>

<Prose>{"The complete "}<code>{"mechanism-calculations.py"}</code>{" also checks the derivative that sharing creates: reshape per-reader value gradients into "}<code>{"[B,Hkv,readers,S,dv]"}</code>{" and sum the readers axis. That result must equal the compact shared V gradient. This is the bridge from a storage choice to actual fitting, not a claim that averaging separately updated MHA weights implements the same update."}</Prose>

<Prose>{"For an independent variation, use six query heads and two KV heads, with value width different from key width. Change the group reshape and output merge consistently; keep the true Q/K scale. "}<strong>{"Hint:"}</strong>{" each KV parameter now receives contributions from three readers. "}<strong>{"Solution:"}</strong>{" SDPA and the direct grouped operator should agree, and three per-reader gradients sum into each shared gradient. The real forecast program already owns projection, optimizer, conversion/uptraining and full-versus-cached state. Reuse it for this extension; do not rebuild the attention derivation or compare unrelated random fits."}</Prose>

<H2>{"6. Convert a trained model and observe what survives"}</H2>

<H3>{"Mean pooling is an initialization, not function preservation"}</H3>

<Prose>{"An MHA checkpoint has separate K/V projection parameters for each head. To initialize a grouped model, average the original K-head matrices inside each new group and do the same for V. Copy the Q maps, output map and other compatible parameters."}</Prose>

<Prose>{"For group "}<InlineMath>{"g"}</InlineMath>{" containing head indices "}<InlineMath>{"I_g"}</InlineMath>{":"}</Prose>

<div className="neural-equation"><MathBlock>{"\\overline W_{K,g}=\\frac1R\\sum_{h\\in I_g}W_{K,h},\\qquad\n\\overline W_{V,g}=\\frac1R\\sum_{h\\in I_g}W_{V,h}."}</MathBlock></div>

<Prose>{"Average K/V biases too when those projections have biases. With PyTorch's output-by-input weight layout, reshape a key weight of shape "}<code>{"[Hq*dk,D]"}</code>{" into "}<code>{"[Hkv,R,dk,D]"}</code>{", average the reader axis and reshape to "}<code>{"[Hkv*dk,D]"}</code>{". Mixing the coordinate axis with the head axis produces a different projection, even if the final tensor has an acceptable shape."}</Prose>

<Prose>{"The mean has a precise limited justification. It minimizes the sum of squared Euclidean/Frobenius distances to the matrices being averaged:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\sum_h\\|W_h-M\\|_F^2\n=\\sum_h\\|W_h-\\overline W\\|_F^2+R\\|M-\\overline W\\|_F^2."}</MathBlock></div>

<Prose>{"The first term is independent of "}<InlineMath>{"M"}</InlineMath>{"; the second is smallest at the mean. This preserves a particular notion of proximity "}<strong>{"in parameter coordinates"}</strong>{". It does not minimize the final model's prediction loss or average attention outputs. Heads can use different learned coordinate bases, and softmax is nonlinear."}</Prose>

<Prose>{"For a fixed query q, comparing it with the mean key gives the mean of its comparisons with all keys in that group. This is not the average of the original heads' own comparisons, which use different queries. Applying softmax then mixing averaged values introduces further nonlinear differences."}</Prose>

<H3>{"A two-head counterexample"}</H3>

<Prose>{"Take scalar queries 1 and 2. Head 0 has keys "}<code>{"[2,0]"}</code>{", values "}<code>{"[1,3]"}</code>{"; head 1 has keys "}<code>{"[0,2]"}</code>{", values "}<code>{"[5,−1]"}</code>{". Their original outputs are approximately 1.238406 and −0.892083. Mean-pooling keys gives "}<code>{"[1,1]"}</code>{", and mean-pooling values gives "}<code>{"[3,1]"}</code>{". Both retained queries now see equal scores over the two positions, so each produces output 2."}</Prose>

<Prose>{"The averaged parameters have not preserved either original output, nor their average. No random numerical accident is needed to demonstrate the issue. Conversely, if the original K and V parameters inside each group are already identical and the positional/masking conventions agree, conversion preserves the function exactly. This tied-head case is a useful null control."}</Prose>

<GqaConversionLab />

<Prose>{"Further training lets the model adapt to the new structure. The original GQA study calls this "}<strong>{"uptraining"}</strong>{". Its T5 experiment continued the pretraining recipe after conversion, then evaluated downstream tasks. It did not establish that every checkpoint can be converted losslessly with a fixed number of updates. Use the original optimization/data recipe when reproducing a reported result; our local study deliberately uses a much smaller and different task."}</Prose>

<H3>{"A real causal forecasting task"}</H3>

<Prose>{"We use the same licensed "}<a href={"https://archive.ics.uci.edu/dataset/181/libras%2Bmovement"}>{"Libras Movement recordings"}</a>{" as the preceding positional lesson, but ask a new question: "}<strong>{"given the observed points so far, where will the next point be?"}</strong>{" Each record has 45 normalized x/y coordinates. Inputs 0–43 predict targets 1–44. The causal mask ensures a prediction cannot read its target or later points. Movement-class labels are used only to preserve the declared split, not as model inputs or forecast targets."}</Prose>

<Prose>{"The source has 360 rows but 330 distinct trajectories. Remove the 30 additional exact duplicates after checking that duplicate labels agree, retain the first occurrence and use the existing seed-73 classwise split: 220 training, 50 validation and 60 test trajectories. Keep whole trajectories together. Fixed scaling "}<InlineMath>{"2x-1"}</InlineMath>{" supplies model inputs and targets; output RMSE is converted back to the original coordinate unit. The source lacks reliable per-row performer/session identifiers, so this is a row-level diagnostic, not evidence of generalization to a new signer or a deployable movement system. "}<a href={"/learn-assets/grouped-query-attention-gqa-multi-query-attention-mqa/data-provenance.md"}>{"Original data, metadata and attribution"}</a>{" accompany the program."}</Prose>

<Prose>{"A persistence baseline predicts the latest observed point unchanged. An affine baseline fits "}<InlineMath>{"\\widehat x_{t+1}=A x_t+b"}</InlineMath>{" to training transitions only. These simple models make it harder to mistake smooth motion for impressive architectural learning."}</Prose>

<H3>{"A controlled conversion protocol"}</H3>

<Prose>{"The neural model has a 2-to-24 coordinate projection, one pre-normalized Transformer block, four query heads of width 6, FFN width 48 with GELU, a final LayerNorm and a two-coordinate prediction at every row. It uses full Q/K RoPE with base 10000 and logical positions 0–43, biased maps, epsilon "}<InlineMath>{"10^{-5}"}</InlineMath>{" and no dropout. It predicts coordinates rather than classes or token probabilities."}</Prose>

<Prose>{"First train the four-KV-head MHA parent using seed 101, 180 full-batch Adam updates at learning rate 0.003. Select the lowest validation MSE, taking the earliest exact tie. The selected parent is update 171."}</Prose>

<Prose>{"From that one checkpoint, create three branches: continue MHA unchanged, convert to two-KV-head GQA by group means, and convert to one-KV-head MQA. Each branch then gets exactly 45 additional full-batch Adam updates at learning rate 0.001 with a fresh optimizer. Select by validation MSE, including the zero-update state as a candidate. The continued MHA branch controls for the benefit of simply doing more optimization. All shared weights are copied from the parent; no branch is retuned after seeing test performance."}</Prose>

<Prose>{"This is a small conversion study, not a comparison of fully optimized architectures trained from scratch. Different branches have different parameter counts and may require different adaptation schedules. The fraction 45/180 is a count of these local updates, not a claim to reproduce the original paper's pretraining-compute fraction."}</Prose>

<H3>{"The actual outcomes"}</H3>

<Prose>{"RMSE below is per coordinate, in the original normalized coordinate units. Each neural test result averages errors over all 44 prediction slots of the same 60 held-out trajectories."}</Prose>

<NeuralTable caption={"The actual outcomes"} headers={[<>{"Model/branch"}</>,<>{"Parameters"}</>,<>{"Selected extra update"}</>,<>{"Test RMSE before extra training"}</>,<>{"Test RMSE after selection"}</>]} rows={[[<>{"MHA continuation, 4 KV heads"}</>,<>{"5,042"}</>,<>{"41"}</>,<>{"0.015789"}</>,<>{"0.015617"}</>],[<>{"Converted GQA, 2 KV heads"}</>,<>{"4,442"}</>,<>{"45"}</>,<>{"0.077182"}</>,<>{"0.018268"}</>],[<>{"Converted MQA, 1 KV head"}</>,<>{"4,142"}</>,<>{"42"}</>,<>{"0.115174"}</>,<>{"0.025224"}</>],[<>{"Persistence baseline"}</>,<>{"0"}</>,<>{"—"}</>,<>{"—"}</>,<>{"0.026458"}</>],[<>{"Training-fitted affine baseline"}</>,<>{"6"}</>,<>{"—"}</>,<>{"—"}</>,<>{"0.026222"}</>]]} />

<Prose>{"Averaging sharply worsens both converted models initially. Further training recovers much of that loss. GQA remains worse than the MHA control in this run; MQA is only slightly better than the simple baselines. These are informative outcomes, not failures to obtain the intended lesson. Cache size reductions are exact consequences of shape; quality recovery is an empirical question."}</Prose>

<Prose>{"Validation RMSE after selection is 0.015260, 0.018438 and 0.025090 for MHA/GQA/MQA respectively. The GQA selection lands on the last allowed update, so this protocol does not establish its eventual converged quality. More adaptation might change the result; do not extend the schedule solely to improve a displayed test score. A new training question would need a newly declared protocol and appropriate evaluation boundary."}</Prose>

<GqaTrainingHistory />

<H3>{"Inspect one real forecast, including the cache"}</H3>

<Prose>{"The visible example was chosen before fitting: source row 77. Observe its first 32 points, then predict point 32 under zero-based numbering. The actual next point is approximately "}<code>{"[0.593810,0.250000]"}</code>{"."}</Prose>

<NeuralTable caption={"Inspect one real forecast, including the cache"} headers={[<>{"Selected branch"}</>,<>{"Next predicted point"}</>,<>{"Compact float32 K/V payload for this 32-point prefix"}</>]} rows={[[<>{"MHA"}</>,<>{"[0.600976,0.258456]"}</>,<>{"6,144 bytes"}</>],[<>{"GQA"}</>,<>{"[0.597412,0.259018]"}</>,<>{"3,072 bytes"}</>],[<>{"MQA"}</>,<>{"[0.607105,0.253203]"}</>,<>{"1,536 bytes"}</>]]} />

<Prose>{"These are actual saved model outputs. The bytes are also checked against the actual K/V tensors: "}<code>{"[1,Hkv,32,6]"}</code>{" for each of K and V. They exclude position metadata and all other model state."}</Prose>

<Prose>{"Feeding those same 32 observed points one at a time with the compact cache reproduced "}<strong>{"every"}</strong>{" full-pass causal forecast within "}<InlineMath>{"2.7\\times10^{-7}"}</InlineMath>{" in transformed coordinates. Shifting all logical IDs by 100 preserved the outputs within "}<InlineMath>{"2.4\\times10^{-7}"}</InlineMath>{" under this fixed RoPE model. Neither check means a new unseen input can be ignored; it verifies equivalence of two computations of the same specified function."}</Prose>

<Prose>{"Reflect observed point 23's x coordinate using "}<InlineMath>{"x\\mapsto1-x"}</InlineMath>{". The final GQA next-point forecast changes from "}<code>{"[0.597412,0.259018]"}</code>{" to "}<code>{"[0.597468,0.258925]"}</code>{"; its earlier per-row predictions can change much more. Causality requires outputs "}<strong>{"before"}</strong>{" the edited point to stay unchanged. Attention to a changed old point can affect later queries even though we use a compact cache. Because the old observation changed, recompute the affected cached states rather than silently reusing a cache for a different prefix."}</Prose>

<GqaForecastLab />

<H3>{"One-step evaluation is not a generated rollout"}</H3>

<Prose>{"In the table, each prediction receives the real preceding observations. This is teacher-forced evaluation. To forecast several future points, feed the model's own output back as its next input. Errors can change subsequent inputs and accumulate."}</Prose>

<Prose>{"Starting from the same 32-point prefix, the GQA model's five generated points are approximately"}</Prose>

<CodeBlock language={"text"}>{"[0.597412,0.259018]\n[0.590012,0.261214]\n[0.583480,0.265978]\n[0.577641,0.272512]\n[0.571721,0.279467]"}</CodeBlock>

<Prose>{"Only the first prediction used the last true observed point as its newest input; each later prediction used the prior prediction. The workspace may reveal the actual recorded future afterward for inspection, but those values must not enter the rollout. Edited trajectories have no newly established ground truth. Also, the linear output is not constrained to 0–1; do not silently clip out-of-range predictions and describe the clipped path as the model's actual output."}</Prose>

<Prose>{"This distinction connects the earlier sequence-model lessons to serving: caching removes repeated computation of an unchanged history; it does not remove the uncertainty or feedback loop of generation."}</Prose>

<H3>{"Reproduce the complete study"}</H3>

<Prose>{"Download "}<a href={"/learn-assets/grouped-query-attention-gqa-multi-query-attention-mqa/author-calculations.py"}>{"the complete CPU program"}</a>{", "}<a href={"/learn-assets/grouped-query-attention-gqa-multi-query-attention-mqa/movement_libras.data"}>{"the data"}</a>{", "}<a href={"/learn-assets/grouped-query-attention-gqa-multi-query-attention-mqa/movement_libras.names"}>{"original metadata"}</a>{" and "}<a href={"/learn-assets/grouped-query-attention-gqa-multi-query-attention-mqa/author-results.json"}>{"the recorded results"}</a>{" into one directory. With Python, NumPy and PyTorch installed:"}</Prose>

<CodeBlock language={"bash"}>{"python author-calculations.py"}</CodeBlock>

<Prose>{"The recorded environment was Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu, one CPU thread and deterministic algorithms. The program includes the complete model, causal/RoPE/grouped operator, mean-weight/bias conversion, both baselines, all fitting and checkpoint selection, split/duplicate checks, actual weight export and full/cached interventions. It does not rely on a hidden notebook, pretrained download or a GPU. Numerical environments can change last digits."}</Prose>

<Prose>{"Read "}<code>{"CausalForecaster.forward"}</code>{" alongside the diagram: normalize the carried state; project native Q/K/V counts; rotate; append compact K/V and logical IDs; reshape the query-head axis into groups/readers; score and mask; mix values; combine heads; complete the residual/FFN path; forecast the next coordinate. The "}<code>{"convert"}</code>{" function shows exactly which parameters are averaged and which are copied."}</Prose>

<GqaProgram />

<Prose>{"The "}<a href={"/learn-assets/grouped-query-attention-gqa-multi-query-attention-mqa/forecast-models.json"}>{"model/trace file"}</a>{" supports later interactive implementation. Its full author evidence is an optional download, not an eager browser payload. A page should load only the selected model's compact weights and selected example, and should evaluate changed inputs without training."}</Prose>

<H2>{"7. Deeper connections: capacity, optimization and deployment"}</H2>

<H3>{"Sharing ties parameters and accumulates gradients"}</H3>

<Prose>{"For a fixed group, several query heads depend on the same K/V parameters. During training, the shared parameters receive contributions from every use. This is the same chain-rule principle as reusing a function or sharing a convolution kernel across image positions."}</Prose>

<Prose>{"If the loss is "}<InlineMath>{"\\mathcal L"}</InlineMath>{" and group g's value at position j is "}<InlineMath>{"v_{g,j}"}</InlineMath>{", its direct value-path derivative is"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{\\partial\\mathcal L}{\\partial v_{g,j}}\n=\\sum_{h:g(h)=g}\\sum_t\na_{h,t,j}\\frac{\\partial\\mathcal L}{\\partial o_{h,t}}."}</MathBlock></div>

<Prose>{"Each reader contributes according to how much it used that value and how its output affected the loss. There is no extra automatic average over group size. If the loss itself is averaged across examples or positions, that normalization is already inside the upstream derivatives."}</Prose>

<Prose>{"Keys have a more coupled effect because they alter softmax weights. For one reader/query, the derivative with respect to its scaled score is"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{\\partial\\mathcal L}{\\partial s_{h,t,j}}\n=a_{h,t,j}\\left(\\frac{\\partial\\mathcal L}{\\partial o_{h,t}}\\right)^\\top\n(v_{g,j}-o_{h,t})."}</MathBlock></div>

<Prose>{"Multiplying by "}<InlineMath>{"q_{h,t}/\\sqrt{d_k}"}</InlineMath>{" and summing over that group's readers/positions gives the shared key's gradient when the key enters scores through this dot product. The subtraction of the current mixture appears because increasing one key's weight decreases other normalized weights. Position transforms add their own chain-rule factors."}</Prose>

<Prose>{"The "}<a href={"/learn-assets/grouped-query-attention-gqa-multi-query-attention-mqa/mechanism-calculations.py"}>{"operator-check program"}</a>{" differentiates a squared-output loss through direct grouped attention and separately through independent repeated value copies. Summing the per-copy gradients exactly recovers the shared-value gradient within float64 tolerance. This is a substantive check of the sharing mechanism, not merely “a gradient exists.”"}</Prose>

<GqaGradientFigure />

<H3>{"What capacity is being restricted?"}</H3>

<Prose>{"At fixed head counts and widths, a grouped model can be represented as an MHA model whose K/V parameters are tied inside each group. MHA allows that tied choice and also permits separate K/V maps. GQA restricts this part of the parameterization while retaining query and output-head diversity."}</Prose>

<Prose>{"This explains a capacity tradeoff, not a theorem that a finite trained MHA model must generalize better. Optimization, data volume, inductive bias and training budgets matter. A restriction can help some tasks, hurt others or make little practical difference. The local forecasting table and published studies are evidence under their own protocols."}</Prose>

<Prose>{"Shared values also do not force all head outputs to be equal: different distributions form different combinations of the same value rows. Before output projection, each head's mixture is in the convex hull of its legal value vectors when attention is ordinary nonnegative row-softmax without dropout. Distinct heads may choose different points in that hull, and their output projections combine them differently. This geometric interpretation does not impose the same convex-hull restriction on the final residual state or forecast."}</Prose>

<Prose>{"If an MHA checkpoint's heads use different feature bases, naive averaging can be a poor functional merge. Even a function-preserving head permutation changes which heads fall into contiguous groups unless grouping is transformed too. Group selection or learned conversion procedures are meaningful research choices. The mean-initialization recipe is a practical baseline, not a proof that the original order yields optimal groups."}</Prose>

<H3>{"What the original studies actually establish"}</H3>

<Prose>{"The "}<a href={"https://arxiv.org/html/2305.13245v3#S3"}>{"GQA paper's Table 1"}</a>{" reports this T5-XXL comparison after its specified 5% uptraining and task fine-tuning:"}</Prose>

<NeuralTable caption={"What the original studies actually establish"} headers={[<>{"Attention"}</>,<>{"Average development-task score"}</>,<>{"Reported inference time, seconds per sample per TPUv4 chip"}</>]} rows={[[<>{"MHA"}</>,<>{"47.2"}</>,<>{"1.51"}</>],[<>{"MQA"}</>,<>{"46.6"}</>,<>{"0.24"}</>],[<>{"GQA with 8 KV groups"}</>,<>{"47.1"}</>,<>{"0.28"}</>]]} />

<Prose>{"The average combines the paper's summarization, translation and question-answering metrics; it is not accuracy on one dataset. The authors optimized parallelization separately and used their stated batching/timing protocol. These rows illustrate their quality/time tradeoff, not a universal GPU speed multiplier. Their Figure 5 varies "}<strong>{"uptraining proportion"}</strong>{"; Figure 6 varies groups and reports "}<strong>{"time"}</strong>{". Neither is a measured seven-point quality curve proving that eight groups are always optimal. The limitations also identify encoder-decoder-only evaluation and the absence of a from-scratch XXL GQA comparison."}</Prose>

<Prose>{"The "}<a href={"https://arxiv.org/pdf/1911.02150"}>{"MQA paper's model-quality section"}</a>{" likewise reports task- and metric-specific differences, including a beam-search translation score where MQA slightly exceeds its MHA baseline. Its comparison widens feedforward layers to match total parameter counts. There is no universal “MQA loses exactly 1%” law. The practical questions are which quality measures matter, what adaptations were performed, and what serving workload was measured."}</Prose>

<H3>{"Three separate ways to reduce attention cost"}</H3>

<NeuralTable caption={"Three separate ways to reduce attention cost"} headers={[<>{"Technique"}</>,<>{"What it changes"}</>,<>{"What it does not automatically establish"}</>]} rows={[[<>{"GQA/MQA"}</>,<>{"Number of distinct K/V heads"}</>,<>{"Fewer legal token positions or fixed-size sequence state"}</>],[<>{"Sliding/local attention"}</>,<>{"Which positions each query can read"}</>,<>{"Identical full-attention function or unlimited direct access to old tokens"}</>],[<>{"Tiled exact attention"}</>,<>{"How the same score/softmax/value calculation is scheduled and stored"}</>,<>{"A different mathematical attention operator or a smaller logical KV cache by itself"}</>],[<>{"Paged cache allocation"}</>,<>{"Where physical cache blocks are stored and shared"}</>,<>{"A change to learned projection widths or a universal throughput multiplier"}</>],[<>{"KV quantization"}</>,<>{"Representation precision and payload packing"}</>,<>{"Exact original floating-point outputs or zero metadata overhead"}</>]]} />

<Prose>{"These can be combined when their implementation contracts agree. "}<a href={"https://arxiv.org/html/2310.06825v1#S2"}>{"Mistral 7B's architecture table"}</a>{" gives a concrete published example with 32 query heads and 8 KV heads, combined with sliding-window attention. That is group size 4, not four KV heads. The window and shared heads act on different axes. Its original context/window choices describe that checkpoint, not a recommendation that every new model use those settings."}</Prose>

<Prose>{"Paged allocation rounds token storage into blocks and can support sharing identical prefix blocks with appropriate reference management. It does not guarantee that a logical eightfold GQA reduction multiplies another claimed fourfold saving into thirty-twofold throughput. Allocation efficiency, bandwidth, scheduling and computation interact. Treat the "}<a href={"https://arxiv.org/abs/2309.06180"}>{"original PagedAttention paper"}</a>{" as further systems reading; the local plots here contain no unmeasured serving numbers."}</Prose>

<Prose>{"Speculative decoding has another role: a draft proposes several tokens and the target verifies them. GQA can supply the target's attention implementation, but its cache layout must support accepted/rejected prefix updates. The benefit depends on acceptance, batch/length and hardware. Do not assign a fixed extra multiplier just because both techniques are present."}</Prose>

<H3>{"Logical sharing can be replicated across devices"}</H3>

<Prose>{"Suppose tensor parallelism partitions query heads across eight devices. A simple implementation may replicate a single MQA KV head on all eight devices so each can perform its local reads. The model logically has one KV head, but aggregate physical storage contains eight copies. With eight GQA KV heads, a suitable one-group-per-device partition can avoid that particular replication."}</Prose>

<Prose>{"For this simplified equal partition, aggregate stored-head copies can be "}<InlineMath>{"\\max(H_{kv},P)"}</InlineMath>{" when head counts and the "}<InlineMath>{"P"}</InlineMath>{" partitions are compatible and smaller counts are replicated. This is an example strategy, not a universal distributed-cache formula. A different algorithm might communicate or partition the state differently. Count actual per-device storage and communications; do not divide every cache estimate by the device count automatically. The "}<a href={"https://arxiv.org/html/2305.13245v3#S2.SS2"}>{"GQA method discussion"}</a>{" explicitly motivates grouped heads partly through this sharding issue."}</Prose>

<GqaDistributedDiagram />

<H3>{"When is this useful beyond chat?"}</H3>

<Prose>{""}<strong>{"Streaming trajectories and sensor prediction."}</strong>{" Our point predictor is a small causal example. A long stream would keep adding cached records unless the architecture or application defines a window, reset or another memory mechanism. Shared heads reduce each stored record's width; they do not make indefinite storage finite. A reliable streaming application also needs a policy for missing observations, changed calibration and corrected historical inputs."}</Prose>

<Prose>{""}<strong>{"Encoder-decoder systems."}</strong>{" A decoder can use GQA for self-attention over generated tokens and for cross-attention over fixed encoder outputs. The cross-attention K/V can be projected once from those encoder outputs and reused while decoding. Its source length can differ from the generated-prefix length. If encoder outputs change, their cached projections must be refreshed. GQA is not restricted to decoder-only language models, although its incremental motivation is strongest in particular workloads."}</Prose>

<Prose>{""}<strong>{"Multimodal prefixes."}</strong>{" Image patches or audio features can contribute many positions to an autoregressive model's context. Shared K/V heads can reduce the representation stored for each such position, but positional coordinates, modality boundaries and legal cross-stream attention remain separate decisions. The later vision and cross-attention lessons own those representations."}</Prose>

<Prose>{""}<strong>{"Mixture-of-experts models."}</strong>{" A Transformer may route feedforward computation to experts while keeping a shared attention layer. In that arrangement, expert count does not multiply the attention KV-head count. Read the actual block definition; the phrase “eight experts” does not mean eight independent copies of every attention cache. "}<a href={"/learn/path/full-curriculum/mixture-of-experts-transformers-moe?module=deep-learning-fundamentals"}>{"Mixture-of-Experts Transformers"}</a>{" develops that distinction."}</Prose>

<GqaOwnershipDiagram />

<H3>{"Choosing or reproducing a configuration"}</H3>

<Prose>{"For a pretrained checkpoint, reproduce its query/KV counts, grouping, widths, positional convention and weights. Changing a configuration field alone is not a conversion. A changed projection shape needs transformed or newly learned parameters and an evaluated adaptation plan."}</Prose>

<Prose>{"For a new model, compare reasonable head counts using the actual task, training budget and serving constraints. A short parallel encoder workload has a different performance profile from long autoregressive decode. Small models can benefit from sharing under suitable contexts, and large models do not all require the same eight groups. Quality, cache capacity, kernel availability and distributed layout jointly determine the choice."}</Prose>

<Prose>{"Measure prefill and decode separately, with stated batch size, sequence lengths, dtype, device, kernels and synchronization. Distinguish latency per request from aggregate tokens per second and account for total model state, not only K/V payload. The transparent NumPy/PyTorch programs here teach correctness. Their Python loops and cache concatenation are not production scheduling advice; repeated concatenation copies existing storage, whereas a bounded or paged implementation can manage append positions directly."}</Prose>

<Prose>{"Finally, the "}<a href={"/learn/path/full-curriculum/multi-head-latent-attention-mla?module=deep-learning-fundamentals"}>{"next topic, Multi-Head Latent Attention"}</a>{", changes a different representation choice: it compresses cache content into a latent representation and carefully handles positional components. Its actual computation cannot be inferred solely from a smaller-looking parameter count. The head/group/cache distinctions here are the prerequisite for understanding that design."}</Prose>

<section className="lesson-ending lesson-ending--practice" data-lesson-ending="practice"><H2>{"8. Practice: reason about new inputs and constraints"}</H2>

<Prose>{"Try each question before opening its hint or solution. Exercises 1–5 check the first-pass route; 6–9 use deeper reasoning. The downloadable programs can verify your calculations, but calculate or explain the mechanism."}</Prose>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"1. Count readers and stored representations"}</H3>

<Prose>{"A model has 12 query heads and 3 KV heads, using contiguous equal groups. Which KV head does query head 9 read? How many distinct attention distributions does one query position produce? If the key at one memory position in group 2 changes, which query heads can be affected? Must all of them change numerically?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Compute the number of readers per group, then apply integer division to the zero-based query-head index."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Group size is "}<InlineMath>{"R=12/3=4"}</InlineMath>{". Head 9 reads group "}<InlineMath>{"\\lfloor9/4\\rfloor=2"}</InlineMath>{". There are 12 attention distributions, one per query head. The changed group-2 key can affect heads 8,9,10,11. A particular score can remain unchanged if its query is orthogonal to the key edit; later output effects can also cancel. Heads outside that group are unaffected by this isolated projected-key change at this operation. A changed original input may influence other projections too, so that is a different intervention."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"2. Sharing does not mean agreement"}</H3>

<Prose>{"Two queries q0="}<code>{"[1,0]"}</code>{" and q1="}<code>{"[0,1]"}</code>{" share keys "}<code>{"[[2,0],[0,2]]"}</code>{" and values "}<code>{"[[1,0],[0,3]]"}</code>{", with head width 2. Compute each distribution and output. Then change the second value to "}<code>{"[2,3]"}</code>{". Which weights change, and by how much do the output x coordinates change?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"The nonzero scaled logit is "}<InlineMath>{"2/\\sqrt2=\\sqrt2"}</InlineMath>{". Let "}<InlineMath>{"p=e^{\\sqrt2}/(1+e^{\\sqrt2})"}</InlineMath>{"."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{""}<InlineMath>{"p\\approx0.804430"}</InlineMath>{". The weights are "}<code>{"[p,1−p]"}</code>{" and "}<code>{"[1−p,p]"}</code>{"; outputs are approximately "}<code>{"[0.804430,0.586711]"}</code>{" and "}<code>{"[0.195570,2.413289]"}</code>{". Changing a value leaves weights unchanged. The x-coordinate increases are "}<InlineMath>{"2(1-p)\\approx0.391141"}</InlineMath>{" and "}<InlineMath>{"2p\\approx1.608859"}</InlineMath>{". Distinct queries read the same value edit in different proportions."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"3. Count bytes without confusing units"}</H3>

<Prose>{"There are two independent requests, 12 layers, 3 KV heads, prefix length 1024, key width 64, value width 32 and two bytes per stored number. Calculate the K/V payload in bytes and MiB. The model has 12 query heads: what would the payload be under MHA with those widths? Name two reasons the measured physical allocation might exceed the compact payload."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Use "}<InlineMath>{"BNLH_{kv}(d_k+d_v)s"}</InlineMath>{". A MiB is "}<InlineMath>{"2^{20}"}</InlineMath>{" bytes. The value width does not have to equal the key width."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The payload is "}<InlineMath>{"2\\times12\\times1024\\times3\\times96\\times2=14,155,776"}</InlineMath>{" bytes, or 13.5 MiB. MHA has four times as many K/V heads, so it needs 54 MiB for these tensors. Reserved/padded capacity, page rounding, quantization metadata, duplicated prefixes or replication across devices can add physical storage; position/cache bookkeeping and other model state also need memory. Do not label this payload as the entire application's memory usage."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"4. Repair a non-square causal mask"}</H3>

<Prose>{"A decode chunk has query positions "}<code>{"[5,6]"}</code>{" and stored key positions "}<code>{"[0,1,2,3,4,5,6]"}</code>{". Write its legal mask. A developer passes only "}<code>{"is_causal=True"}</code>{" to an API documented to use an upper-left triangle for non-square inputs. What relation does that represent instead? Would GQA head sharing fix the error?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Compare actual logical positions, then separately consider local row indices 0 and 1."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The correct rows are "}<code>{"[1,1,1,1,1,1,0]"}</code>{" and "}<code>{"[1,1,1,1,1,1,1]"}</code>{". An upper-left triangle gives "}<code>{"[1,0,0,0,0,0,0]"}</code>{" and "}<code>{"[1,1,0,0,0,0,0]"}</code>{", as if queries were at local positions 0 and 1. Supply the intended explicit mask or an API-specific offset-aware causal bias. The head axis and causal-position axis are independent; fewer KV heads cannot repair the wrong legal set."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"5. Design a fair conversion check"}</H3>

<Prose>{"After mean-pooling an MHA checkpoint, a researcher trains only the converted model for 100 more updates, then compares it with the original checkpoint. What extra control would help separate conversion recovery from additional training? Which data can select the checkpoint? Under what special condition is mean conversion exactly function-preserving before any further update?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Match the continuation budget, keep test data outside selection, and think about already-tied projection heads."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Continue an unchanged MHA branch for the same declared additional-update protocol, and report both models' before/after states. Use validation data for checkpoint selection and held-out test data for final assessment, with group/duplicate boundaries appropriate to the task. Mean conversion preserves the function if original K/V maps, including biases, are identical inside each target group and all relevant positional, mask and output conventions remain consistent. General group averaging is not lossless. Equal update counts also do not necessarily mean equal FLOPs, so state the comparison budget honestly."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"6. Accumulate a shared value gradient"}</H3>

<Prose>{"Two readers attend to one shared scalar value with weights 0.2 and 0.7 at the selected memory position. Their upstream output derivatives are 3 and −1. There is one query position per reader, and the loss already contains any intended averaging. What is the shared value gradient from these two uses? Would dividing by group size be correct?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Each use contributes attention weight times its upstream derivative. Add the contributions."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The gradient is "}<InlineMath>{"0.2\\times3+0.7\\times(-1)=-0.1"}</InlineMath>{". The contributions partly cancel. An extra division by two would change the defined objective's gradient; sharing sums uses. Any averaging desired by the loss must be specified in that loss and propagated normally."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"7. A cache ratio is not a timing result"}</H3>

<Prose>{"In an explicitly assumed timing model, 40% of a step is work that would become four times faster, while the remaining 60% is unchanged. Calculate the total speedup. If compact KV storage also falls fourfold, may you report the calculated time as measured GPU performance?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Normalize the original time to 1 and add the new times of the two parts."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"New time is "}<InlineMath>{"0.6+0.4/4=0.7"}</InlineMath>{", so speedup is "}<InlineMath>{"1/0.7\\approx1.429"}</InlineMath>{". It is a consequence of the assumed decomposition, not a measurement. Real kernels may change reuse, occupancy and communication, and the fourfold storage reduction does not establish that the affected time portion accelerates fourfold. Label modeled and measured quantities separately."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"8. Account for distributed replicas"}</H3>

<Prose>{"A model has 32 query heads, 2 KV heads and 8 tensor-parallel devices. Under a simple layout that puts four queries on each device and replicates each KV head on the four devices reading it, how many physical KV-head copies exist in aggregate? How does that compare with its logical head count? Give another layout choice whose cost would need separate analysis."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Each of the two logical KV heads has four physical copies. Avoid assuming that model parallelism always divides every tensor evenly."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"There are "}<InlineMath>{"2\\times4=8"}</InlineMath>{" physical KV-head copies, four times the logical count. The local query computations remain distinct, but the shared memory has been replicated. A communicating or differently sharded cache could reduce replication while adding communication or another scheduling constraint. The correct accounting depends on that algorithm, not on head count alone."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"9. Interpret a forecast without leaking its future"}</H3>

<Prose>{"An observed prefix ends at point 19. You want five future predictions. A program computes the first forecast, then feeds the true point 20 before computing its second forecast. Has it performed a five-step generated rollout? If a learner edits observed point 7, can the old cache safely be reused unchanged? Explain both dependency errors."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Identify what information is available at the observation boundary, then what the cache's old states were functions of."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Feeding true point 20 makes the later calculation teacher-forced, not an autonomous rollout from the original boundary. A generated rollout feeds the model's own forecast as the next input. Editing point 7 changes projections and, in a multilayer causal network, can affect later hidden states and caches. Recompute the affected prefix states or use a correctly designed invalidation strategy; an old cache is valid for its original prefix, model and position convention. Caching accelerates an unchanged computation, not a changed history."}</Prose>

</details></div></section>

<section className="lesson-ending lesson-ending--next" data-lesson-ending="next"><H2>{"What comes next?"}</H2>

<Prose>{"You can now distinguish query diversity from stored-head diversity, trace exact grouping and masks, count compact cache payload and evaluate what conversion actually preserves. Continue to "}<a href={"/learn/path/full-curriculum/multi-head-latent-attention-mla?module=deep-learning-fundamentals"}>{"Multi-Head Latent Attention"}</a>{". It compresses the cached representation through learned latent coordinates, so its score/value reconstruction and rotary components need a new derivation rather than a relabeled head-count diagram."}</Prose></section>

<section className="lesson-ending lesson-ending--resources" data-lesson-ending="resources"><H2>{"References and another way to learn"}</H2>

<ul><li>{""}<strong>{"Original mechanism and bandwidth reasoning:"}</strong>{" "}<a href={"https://arxiv.org/pdf/1911.02150"}>{"Shazeer, Fast Transformer Decoding: One Write-Head is All You Need"}</a>{". The full nine-page source was inspected, including batched/incremental "}<code>{"einsum"}</code>{" definitions, MQA construction, performance assumptions and translation/language-model protocols. Read §§2–3 for the computational argument, then §4 to see why a particular measured speedup and a particular quality metric must stay attached to their experiment."}</li><li>{""}<strong>{"Grouping and checkpoint conversion:"}</strong>{" "}<a href={"https://arxiv.org/html/2305.13245v3"}>{"Ainslie et al., GQA"}</a>{". The complete methods, experiments, ablations, limitations and stability appendix were inspected. The section structure is a useful deeper reading route: mean conversion, continued training, grouped heads, then actual comparisons. The "}<a href={"https://aclanthology.org/2023.emnlp-main.298.mp4"}>{"official EMNLP presentation"}</a>{" is linked from the "}<a href={"https://aclanthology.org/2023.emnlp-main.298/"}>{"ACL paper record"}</a>{". Its provenance and associated full paper were checked; the video was not independently watched or transcribed for this packet. Use it as an optional author presentation, not as separate verified experimental evidence."}</li><li>{""}<strong>{"A short alternative explanation:"}</strong>{" "}<a href={"https://sebastianraschka.com/llms-from-scratch/ch04/04_gqa/"}>{"Sebastian Raschka's GQA guide"}</a>{". The full inspected guide connects shared heads, smaller cache state and combinations with other architecture choices. It is a compact conceptual recap rather than this lesson's detailed conversion/masking proof. No popularity statement in that guide substitutes for a checkpoint's actual architecture disclosure."}</li><li>{""}<strong>{"An illustrated attention prerequisite:"}</strong>{" "}<a href={"https://www.3blue1brown.com/lessons/attention/"}>{"3Blue1Brown's Attention in transformers, step-by-step"}</a>{" has a video and written adaptation. The Q/K/softmax/value-mixing portions were inspected during the preceding attention/position packets and are reused as background reading. It explains why different queries can read shared information differently; it does not teach GQA cache implementation. Its visual orientation may use query columns rather than this lesson's query rows."}</li><li>{""}<strong>{"Versioned operator reference:"}</strong>{" "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html"}>{"PyTorch 2.14 scaled dot-product attention"}</a>{". The actual shape, grouping, dropout, boolean-mask, non-square-causal and backend sections were inspected. Match the documentation to the version you run. "}<a href={"/learn-assets/grouped-query-attention-gqa-multi-query-attention-mqa/mechanism-calculations.py"}>{"Our complete native checks"}</a>{" record what was executed locally."}</li><li>{""}<strong>{"Read a production attention layer:"}</strong>{" "}<a href={"https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py"}>{"Transformers' Llama attention source"}</a>{". The projection, native-head rotation, cache update, repeat helper and backend dispatch were inspected. The branch is mutable; pin a version for reproduction. The lesson's complete model is intentionally much smaller and independently implemented."}</li><li>{""}<strong>{"Kernel layout and mask conventions:"}</strong>{" "}<a href={"https://github.com/Dao-AILab/flash-attention#how-to-use-flashattention"}>{"FlashAttention usage and cache documentation"}</a>{". The GQA head mapping, unequal-length causal alignment, cache update and rotary-layout contracts were inspected. These details are valuable after the transparent implementation; no FlashAttention installation or GPU timing was performed here."}</li><li>{""}<strong>{"A concrete combined architecture:"}</strong>{" "}<a href={"https://arxiv.org/html/2310.06825v1#S2"}>{"Mistral 7B §2"}</a>{". The actual architecture table, rolling-cache and prefill/chunking explanation show that GQA and a local window address different dimensions. Read the checkpoint's own scope rather than generalizing its reported outcomes to every context or model."}</li><li>{""}<strong>{"Offline real-data work:"}</strong>{" "}<a href={"https://archive.ics.uci.edu/dataset/181/libras%2Bmovement"}>{"UCI Libras Movement"}</a>{", "}<a href={"/learn-assets/grouped-query-attention-gqa-multi-query-attention-mqa/data-provenance.md"}>{"data/protocol provenance"}</a>{", "}<a href={"/learn-assets/grouped-query-attention-gqa-multi-query-attention-mqa/author-calculations.py"}>{"complete forecasting/conversion program"}</a>{", "}<a href={"/learn-assets/grouped-query-attention-gqa-multi-query-attention-mqa/author-results.json"}>{"recorded outcomes"}</a>{" and "}<a href={"/learn-assets/grouped-query-attention-gqa-multi-query-attention-mqa/mechanism-fixtures.json"}>{"independent fixtures"}</a>{". The small study makes conversion and actual cached outputs inspectable without a large-model download."}</li></ul></section>
</div> };
export default lesson;
